"""
train_kc_mastery.py — 训练 Per-KC Mastery 模型

用法:
  python train_kc_mastery.py --data_dir Code-S --emb_size 64 --batch_size 12

相比 train_mlkt4.py 的改动:
  1. 使用 PerKCMasteryModel (替代 StudentKnowledgeModel)
  2. 去掉 use_debug 参数 (新模型不包含纠错能力)
  3. 因为是标量掌握度, emb_size 可以更小 (64)
"""

import sys
import pandas as pd
import argparse
import os
import json

# [5] CUDA调试开关: 默认关闭(训练快), 需要精确定位CUDA错误时设 KT_DEBUG_CUDA=1
if os.environ.get('KT_DEBUG_CUDA'):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
import numpy as np
import torch
from pytorchtools import EarlyStopping
from mlkt_utils import generate_time_intervals_and_write_back
import tqdm
import torch.optim as optim
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from gen_data import KTDataset
from mlkt4_kc_mastery import VAEPerKCMasteryModel
from eval_mlkt import performance
from torch.utils.data import DataLoader

seed = 42
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Train Per-KC Mastery Model')
# train params
parser.add_argument('--max_iter', type=int, default=500, help='number of iterations')
parser.add_argument('--seed', type=int, default=224, help='default seed')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--max_length', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=12)

# dataset params
parser.add_argument('--data_dir', type=str, default='Code-S', choices=['Code-S', 'Code-F', 'BePKT'])
parser.add_argument('--model', type=str, default='MSDGKT', choices=['MSDGKT'])  # 固定MSDGKT, 因为使用MSDGKT目录

# model params
parser.add_argument('--device', type=str, default=device, help='device')
parser.add_argument('--emb_size', type=int, default=64, help='embedding size (default 64 for scalar mastery)')

# 实验参数
parser.add_argument('--fold', type=int, default=-1, help='只跑单个fold(-1=全部5折), 用于快速实验试水')
parser.add_argument('--kc_mode', type=str, default='kcmask', choices=['kcmask', 'skills'],
                    help='知识组件来源: kcmask=AST-KC自动发现(默认), skills=人工Skills')

params = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==== 动态读取知识组件数量 (kc_mode 决定来源) ====
KC_MAPPING_PATH = os.path.join(BASE_DIR, '..', '..', 'data', params.data_dir, 'ast_kc_mapping.json')
if params.kc_mode == 'kcmask':
    if os.path.exists(KC_MAPPING_PATH):
        with open(KC_MAPPING_PATH, 'r', encoding='utf-8') as f:
            kc_mapping = json.load(f)
        NUM_SKILLS = kc_mapping['num_kcs']
        print(f'[Per-KC Mastery] 使用 {NUM_SKILLS} 个自动KCs (kc_mode=kcmask)')
    else:
        raise FileNotFoundError(f'需要KC映射文件: {KC_MAPPING_PATH}')
else:
    # skills 模式: 从 q_a 的 Skills 列推断人工skill数量 (最大索引+1)
    qa_path = os.path.join(BASE_DIR, '..', '..', 'data', params.data_dir, params.model, 'q_a_train_detailed.csv')
    qa_df = pd.read_csv(qa_path)
    max_skill = max(max(eval(s)) for s in qa_df['Skills'])
    NUM_SKILLS = max_skill + 1
    print(f'[Per-KC Mastery] 使用 {NUM_SKILLS} 个人工Skills (kc_mode=skills)')

if params.data_dir in ['Code-S', 'Code-F']:
    NUM_QUESTIONS = 50
elif params.data_dir in ['BePKT']:
    NUM_QUESTIONS = 553

# 数据已有TimeInterval列 (由data_save生成), 无需再调用mlkt_utils
# [7] 数据集/DataLoader 移到每折内创建, 以便读取 fold{fold_idx}/ 数据划分 (真5折)

# Log file
save_log = './log'
if not os.path.exists(save_log):
    os.makedirs(save_log)
# [A/B] 实验标签: 默认(base)文件名不变; 增强版加后缀区分
_ab_tag = ''
log_file = os.path.join(
    save_log + f'/KCMastery_d{params.emb_size}_{params.data_dir}{_ab_tag}_test_result.txt'
)
log = open(log_file, 'w')

rmse_test_list = []
mae_test_list = []
acc_test_list = []
auc_test_list = []
now = time.time()

folds_to_run = [params.fold] if params.fold >= 0 else range(5)
print(f'运行折数: {list(folds_to_run)}')
for fold_idx in folds_to_run:
    print(f'\n{"="*60}')
    print(f'Fold {fold_idx}')
    print(f'{"="*60}')
    early_stopping = EarlyStopping(patience=10, verbose=True)

    save_model_file = os.path.join(
        f'./fold{fold_idx}/{params.data_dir}/KCMastery_d{params.emb_size}{_ab_tag}'
    )
    os.makedirs(save_model_file, exist_ok=True)

    # [7] 每折独立显式种子 (可复现; 真5折下每折是不同数据划分+不同初始化)
    torch.manual_seed(seed + fold_idx)
    np.random.seed(seed + fold_idx)

    # [7] 每折加载对应数据划分 (KTDataset 按 fold{fold_idx} 读数据)
    params.fold = fold_idx
    train_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Train')
    valid_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Valid')
    test_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Test')
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    # ---- 创建 VAE 模型 (单步KL) ----
    model = VAEPerKCMasteryModel(
        NUM_QUESTIONS, NUM_SKILLS, params.emb_size, K=8,
        device=params.device, beta=0.001
    ).to(params.device)

    # 加载KC邻接矩阵 (图卷积)
    if os.path.exists(KC_MAPPING_PATH):
        with open(KC_MAPPING_PATH, 'r', encoding='utf-8') as f:
            kc_data = json.load(f)
        if 'kc_adjacency' in kc_data:
            adj = torch.tensor(kc_data['kc_adjacency'], dtype=torch.float, device=params.device)

            model.kc_adj = adj
            edge_count = int(adj.sum().item() - NUM_SKILLS)
            print(f"  加载KC图: {NUM_SKILLS}节点, {edge_count}条边")

    model_adam = optim.Adam(model.parameters(), lr=params.lr)

    # ---- 训练 ----
    for epoch in range(params.max_iter):
        print(f'epoch {epoch+1}')
        model.set_epoch(epoch)  # 更新KL退火
        model.train()
        for batch in tqdm.tqdm(train_dataloader, desc='Training'):
            problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor, debug_tensor = batch

            pred_score, kl_step, extra = model(
                problem_tensor, skills_tensor, label_tensor,
                reason_tensor, time_tensor[:, 1:],
                global_debug=debug_tensor
            )
            pad_mask = (note_tensor[:, 1:] != -1).float()
            pred_loss, bce_val, kl_val = model.compute_loss(
                pred_score[:, 1:], label_tensor[:, 1:], kl_step, mask=pad_mask
            )

            if torch.isnan(pred_loss) or torch.isinf(pred_loss):
                print(f'  ⚠️ 跳过batch: loss={pred_loss.item():.4f}')
                model_adam.zero_grad()
                continue

            model_adam.zero_grad()
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_adam.step()

        # ---- 验证 ----
        model.eval()
        all_preds = []
        all_labels = []
        bce_list = torch.Tensor([]).to(params.device)
        kl_list = torch.Tensor([]).to(params.device)
        with torch.no_grad():
            for batch in tqdm.tqdm(valid_dataloader, desc='Validating'):
                problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor, debug_tensor = batch
                pred, kl_step, extra = model(
                    problem_tensor, skills_tensor, label_tensor,
                    reason_tensor, time_tensor[:, 1:],
                    global_debug=debug_tensor
                )
                all_preds.append(pred[:, 1:])
                all_labels.append(note_tensor[:, 1:])
                pad_mask = (note_tensor[:, 1:] != -1).float()
                # [1] 早停监控验证BCE(不含KL), 模型选择对齐预测质量
                loss, bce_v, kl_v = model.compute_loss(pred[:, 1:], label_tensor[:, 1:], kl_step, mask=pad_mask)
                bce_list = torch.cat((bce_list, torch.tensor([bce_v], device=params.device)), dim=0)
                kl_list = torch.cat((kl_list, torch.tensor([kl_v], device=params.device)), dim=0)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        valid_rmse, valid_mae, valid_acc, valid_auc = performance(all_labels, all_preds)

        bce_mean = bce_list.mean().item()
        kl_mean = kl_list.mean().item()
        # [4] 诊断量 + 日志立即落盘
        ar = min(1.0, epoch / 20.0)   # 与模型 compute_loss 的退火一致
        beta_ar_kl = model.beta * ar * kl_mean
        print(f'  valid_rmse: {valid_rmse:.4f}  valid_mae: {valid_mae:.4f}  valid_acc: {valid_acc:.4f}  valid_auc: {valid_auc:.4f}')
        print(f'  [diag] bce={bce_mean:.4f} kl={kl_mean:.2f} beta*ar*kl={beta_ar_kl:.4f}', flush=True)
        print(f'  [diag] bce={bce_mean:.4f} kl={kl_mean:.2f} beta*ar*kl={beta_ar_kl:.4f}', file=log, flush=True)

        # [1] 早停用验证BCE(不含KL)
        early_stopping(bce_list.mean(), model,
                       save_path=os.path.join(save_model_file, 'kt_model_best.pt'))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # ---- 测试 (固定0.5阈值, 不用阈值校准) ----
    load_model_path = os.path.join(save_model_file, 'kt_model_best.pt')
    model.load_state_dict(torch.load(load_model_path, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, desc='Testing'):
            problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor, debug_tensor = batch
            pred, _, _ = model(
                problem_tensor, skills_tensor, label_tensor,
                reason_tensor, time_tensor[:, 1:],
                global_debug=debug_tensor
            )
            all_preds.append(pred[:, 1:])
            all_labels.append(note_tensor[:, 1:])

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        rmse, mae, acc, auc = performance(all_labels, all_preds)
        rmse_test_list.append(rmse)
        mae_test_list.append(mae)
        acc_test_list.append(acc)
        auc_test_list.append(auc)
        print(f'  test_rmse: {rmse:.4f}  test_mae: {mae:.4f}  test_acc: {acc:.4f}  test_auc: {auc:.4f}')
        print(f'  test_rmse: {rmse:.4f}  test_mae: {mae:.4f}  test_acc: {acc:.4f}  test_auc: {auc:.4f}', file=log)

# ---- 最终结果 ----
print(f'\n{"="*60}')
print(f'Per-KC Mastery 最终结果 (emb_size={params.emb_size}, {params.data_dir})')
print(f'{"="*60}')
print(f'RMSE: {np.mean(rmse_test_list):.4f} ± {np.std(rmse_test_list):.4f}')
print(f'MAE:  {np.mean(mae_test_list):.4f} ± {np.std(mae_test_list):.4f}')
print(f'ACC:  {np.mean(acc_test_list):.4f} ± {np.std(acc_test_list):.4f}')
print(f'AUC:  {np.mean(auc_test_list):.4f} ± {np.std(auc_test_list):.4f}')
print(f'总耗时: {(time.time()-now)/60:.2f} min')

print(f'\n{"="*60}', file=log)
print(f'Per-KC Mastery 最终结果 (emb_size={params.emb_size}, {params.data_dir})', file=log)
print(f'{"="*60}', file=log)
print(f'RMSE: {np.mean(rmse_test_list):.4f} ± {np.std(rmse_test_list):.4f}', file=log)
print(f'MAE:  {np.mean(mae_test_list):.4f} ± {np.std(mae_test_list):.4f}', file=log)
print(f'ACC:  {np.mean(acc_test_list):.4f} ± {np.std(acc_test_list):.4f}', file=log)
print(f'AUC:  {np.mean(auc_test_list):.4f} ± {np.std(auc_test_list):.4f}', file=log)
log.close()
