import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

def performance(ground_truth, prediction, threshold=0.5):
    # 将输入展平为一维数组
    ground_truth = ground_truth.flatten().detach().cpu().numpy()  # 移动到 CPU 并转换为 NumPy
    prediction = prediction.flatten().detach().cpu().numpy()  # 移动到 CPU 并转换为 NumPy

    # 过滤掉 ground_truth 中值为 -1 的无效数据
    valid_indices = ground_truth != -1
    if not valid_indices.any():
        return 0.0, 0.0, 0.0, 0.0  # 返回默认值，避免空数据错误

    ground_truth = ground_truth[valid_indices]
    prediction = prediction[valid_indices]

    # 计算 MSE
    MSE = mean_squared_error(ground_truth, prediction)

    # 计算 RMSE
    RMSE = np.sqrt(MSE)

    # 计算 MAE
    MAE = np.mean(np.abs(ground_truth - prediction))

    # 将连续值二值化（以 threshold 为阈值）用于计算 ACC
    # threshold 可由验证集搜索得到 (find_best_threshold); 默认 0.5
    pred_binary = (prediction >= threshold).astype(int)
    true_binary = (ground_truth >= 0.5).astype(int)

    # 计算 ACC
    ACC = accuracy_score(true_binary, pred_binary)

    # 计算 AUC（需要正负样本都存在）
    if len(np.unique(true_binary)) > 1:
        AUC = roc_auc_score(true_binary, prediction)
    else:
        AUC = 0.0

    return RMSE, MAE, ACC, AUC


def find_best_threshold(ground_truth, prediction, th_min=0.3, th_max=0.7, step=0.01):
    """在验证集上搜索使ACC最大的二值化阈值

    Args:
        ground_truth: 真实分数 (连续, >=0.5为对)
        prediction: 预测概率
    Returns:
        (best_threshold, best_acc): 最优阈值及其ACC
    """
    gt = ground_truth.flatten().detach().cpu().numpy()
    pr = prediction.flatten().detach().cpu().numpy()
    valid = gt != -1
    gt, pr = gt[valid], pr[valid]
    if not valid.any():
        return 0.5, 0.0
    tb = (gt >= 0.5).astype(int)
    best_acc, best_th = 0.0, 0.5
    for th in np.arange(th_min, th_max + step, step):
        acc = accuracy_score(tb, (pr >= th).astype(int))
        if acc > best_acc:
            best_acc, best_th = acc, th
    return best_th, best_acc

class lossFunc(nn.Module):
    def __init__(self, device):
        super(lossFunc, self).__init__()
        self.device = device
        self.bce_loss = nn.BCELoss(reduction='none')  # 分数预测的二元交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # 原因预测的交叉熵损失

    def forward(self, pred, label, note, reason_tensor=None, pred_reason_logic=None, pred_reason_concept=None,logic_reason_weight=None,concept_reason_weight=None):
        # 移动张量到指定设备
        note = note.to(self.device)
        label = label.to(self.device)

        # 创建掩码，标记有效标签（note != -1 的位置）
        mask = (note != -1).float()  # [B, S-1]
        valid_count = mask.sum() + 1e-8  # 有效位置数量，防止除零

        # 计算分数预测的损失
        # [fix] BCE = -(t·log x + (1-t)·log(1-x)) 的输入必须落在 (0,1) 开区间:
        #   ① clamp(NaN) 仍是 NaN → 先检测+nan_to_num;
        #   ② 原 clamp(0,1) 允许 x=1 / x=0, 对应梯度为 1e12 量级(而 mask=0 的 padding 位置
        #      正是大量饱和到 1.0 的地方), 且 CUDA 的 BCE kernel 会断言 0<=input<=1;
        #   ③ per-KC 模型的 compute_loss 早已用 clamp(1e-7, 1-1e-7), 此处与之对齐。
        n_bad_pred = int((~torch.isfinite(pred)).sum())
        if n_bad_pred:
            print(f'  [loss][!] pred 含 {n_bad_pred} 个非有限值(已替换): '
                  f'nan={int(torch.isnan(pred).sum())} inf={int(torch.isinf(pred).sum())} '
                  f'-> 上游参数/状态已出现 NaN/Inf, 请用 --nan_debug 定位')
        pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0)
        pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)
        label = label.float()
        n_bad_label = int((~torch.isfinite(label)).sum())
        if n_bad_label:
            print(f'  [loss][!] label 含 {n_bad_label} 个非有限值(已替换为 0)')
        label = torch.clamp(torch.nan_to_num(label, nan=0.0), min=0.0, max=1.0)
        score_loss = self.bce_loss(pred, label)  # [B, S-1]
        masked_score_loss = score_loss * mask
        final_score_loss = masked_score_loss.sum() / valid_count

        if reason_tensor is None:
            # 仅返回分数损失
            return final_score_loss

        # 移动原因相关张量
        reason_tensor = reason_tensor.to(self.device)
        pred_reason_logic = pred_reason_logic.to(self.device)
        pred_reason_concept = pred_reason_concept.to(self.device)

        # 拆分 reason_tensor
        # err_reason 顺序: [逻辑正确，语法正确，粗心导致的语法错误，不熟练导致的语法错误，逻辑错误]
        logic_reason_tensor = torch.stack([reason_tensor[:, :, 0], reason_tensor[:, :, 4]], dim=-1)  # [B, S-1, 2]
        concept_reason_tensor = reason_tensor[:, :, 1:4]  # [B, S-1, 3]

        # 计算逻辑原因损失
        B, S_minus_1, _ = logic_reason_tensor.shape
        pred_reason_logic_flat = pred_reason_logic.contiguous().view(-1, 2)  # [B*(S-1), 2]
        logic_reason_labels = torch.argmax(logic_reason_tensor.view(-1, 2), dim=-1)  # [B*(S-1)]
        logic_reason_loss = self.ce_loss(pred_reason_logic_flat, logic_reason_labels)  # [B*(S-1)]
        logic_reason_loss = logic_reason_loss.view(B, S_minus_1)  # [B, S-1]
        masked_logic_reason_loss = logic_reason_loss * mask
        final_logic_reason_loss = masked_logic_reason_loss.sum() / valid_count

        # 计算概念原因损失
        pred_reason_concept_flat = pred_reason_concept.contiguous().view(-1, 3)  # [B*(S-1), 3]
        concept_reason_labels = torch.argmax(concept_reason_tensor.view(-1, 3), dim=-1)  # [B*(S-1)]
        concept_reason_loss = self.ce_loss(pred_reason_concept_flat, concept_reason_labels)  # [B*(S-1)]
        concept_reason_loss = concept_reason_loss.view(B, S_minus_1)  # [B, S-1]
        masked_concept_reason_loss = concept_reason_loss * mask
        final_concept_reason_loss = masked_concept_reason_loss.sum() / valid_count
        # print('final_score_loss:',final_score_loss.item())
        # print('final_logic_reason_loss:',final_logic_reason_loss)
        # print('final_concept_reason_loss:',final_concept_reason_loss)
        # 总损失
        total_loss = final_score_loss + 0.01*final_logic_reason_loss + 0.01*final_concept_reason_loss

        return total_loss