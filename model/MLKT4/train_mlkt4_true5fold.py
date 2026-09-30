# -*- coding: utf-8 -*-
"""
train_mlkt4_true5fold.py
=========================
在 6:2:2 学生级真5折数据上训练 mlkt4 主模型 (StudentKnowledgeModel, model=MLKT4, 无debug分支),
作为 Per-KC Mastery (VAE) 的公平对比基线。

与 train_mlkt4.py 的差异:
  [7] 真5折: 每折从 data/BePKT/MSDGKT/fold{fold_idx}/ 读取对应 q_a_{train,valid,test}_detailed.csv
       (配套 _make_folds.py + gen_data.py 的 fold 支持)
  [7] 每折独立种子: seed + fold_idx
  [7] kc_mode: 默认 kcmask, 与 Per-KC 输入一致 (读 KC_mask 列 = 同一份 AST-KC)
  [5] CUDA_LAUNCH_BLOCKING 默认关闭(提速), 需要精确定位时设 KT_DEBUG_CUDA=1
  [x] 不再调用 generate_time_intervals_and_write_back (fold 数据已带 TimeInterval)

用法 (服务器, cuda):
  python train_mlkt4_true5fold.py --data_dir BePKT --model MLKT4 \
         --emb_size 256 --batch_size 12 --max_iter 500 --fold -1
  # 先验证单折:
  python train_mlkt4_true5fold.py --data_dir BePKT --model MLKT4 --emb_size 256 \
         --batch_size 12 --max_iter 50 --fold 0

输出: log/wo_ReasonLossMLKT4_d256_BePKT_true5fold_test_result.txt
"""
import sys
import random
import pandas as pd
import argparse
import os
import json

if os.environ.get('KT_DEBUG_CUDA'):   # [5] 默认关, 需要精确定位CUDA错误时设 1
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
import numpy as np
import torch
from pytorchtools import EarlyStopping
import tqdm
import torch.optim as optim
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from gen_data import KTDataset
from mlkt4 import StudentKnowledgeModel
from eval_mlkt import lossFunc, performance
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Train MLKT4 on true 5-fold (6:2:2)')
parser.add_argument('--max_iter', type=int, default=500)
parser.add_argument('--nan_debug', action='store_true',
                    help='开启 autograd anomaly detection: 抛出 NaN/Inf 产生的具体算子(定位用, 很慢)')
parser.add_argument('--seed', type=int, default=224)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_length', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--data_dir', type=str, default='BePKT', choices=['Code-S', 'Code-F', 'BePKT'])
parser.add_argument('--model', type=str, default='MLKT4', choices=['MLKT4', 'MSDGKT'],
                    help='MLKT4=纯mlkt4(use_debug=False,推荐作基线); MSDGKT需要GlobalDebugAbility列(fold数据没有→退化为0)')
parser.add_argument('--kc_mode', type=str, default='kcmask', choices=['kcmask', 'skills'],
                    help='kcmask=读KC_mask列(AST-KC,与Per-KC输入一致); skills=人工Skills列')
parser.add_argument('--fold', type=int, default=-1, help='>=0 跑单折; -1 跑全部5折')
parser.add_argument('--device', type=str, default=device)
parser.add_argument('--emb_size', type=int, default=256)

params = parser.parse_args()

# ==== 解耦: --model 只决定模型结构(use_debug), 数据子目录固定用 MSDGKT ====
# (fold 数据存放在 data/<data_dir>/MSDGKT/foldN/ 下, 与模型类型无关;
#  KTDataset 用 args.model 拼数据路径, 若不重定向会去读 data/<dir>/MLKT4/ → FileNotFound)
model_tag = params.model                    # 'MLKT4'(无debug,推荐基线) 或 'MSDGKT'
USE_DEBUG = (model_tag == 'MSDGKT')
params.model = 'MSDGKT'                     # 数据子目录固定 (fold数据所在)

# ==== 概念/KC 数量 (按 kc_mode 决定; 必须与数据里的 c_id 上界一致, 否则 Embedding 越界) ====
KC_MAPPING_PATH = f'../../data/{params.data_dir}/ast_kc_mapping.json'


def _max_concept_id(data_dir, col):
    """读数据该列的最大 id (优先 fold0 的 train/valid/test = 全量; 否则非折目录)"""
    base = f'../../data/{data_dir}/MSDGKT'
    paths = [os.path.join(base, 'fold0', f'q_a_{s}_detailed.csv') for s in ['train', 'valid', 'test']]
    if not all(os.path.exists(p) for p in paths):
        paths = [os.path.join(base, f'q_a_{s}_detailed.csv') for s in ['train', 'valid', 'test']]
    mx = -1
    for p in paths:
        if os.path.exists(p):
            for s in pd.read_csv(p, usecols=[col])[col]:
                v = eval(s) if isinstance(s, str) else s
                if len(v):
                    mx = max(mx, max(v))
    return mx


if params.kc_mode == 'kcmask':
    if os.path.exists(KC_MAPPING_PATH):
        with open(KC_MAPPING_PATH, 'r', encoding='utf-8') as f:
            kc_mapping = json.load(f)
        NUM_SKILLS = kc_mapping['num_kcs']
        print(f'[KC模式] kcmask: 使用 {NUM_SKILLS} 个AST-KCs (来自 {KC_MAPPING_PATH})')
    else:
        MAX_CID = _max_concept_id(params.data_dir, 'KC_mask')
        NUM_SKILLS = MAX_CID + 1
        print(f'[KC模式] kcmask: 无KC映射文件 → 从数据推用 {NUM_SKILLS} 个AST-KCs (max id={MAX_CID})')
else:
    # skills: 概念数必须从数据的 Skills 列推 (BePKT=106), 不能沿用 kc_mapping['num_kcs']=69
    MAX_CID = _max_concept_id(params.data_dir, 'Skills')
    NUM_SKILLS = MAX_CID + 1
    print(f'[KC模式] skills: 数据中最大概念 id={MAX_CID} → 使用 {NUM_SKILLS} 个人工Skills')

if params.data_dir in ['Code-F', 'Code-S']:
    NUM_QUESTIONS = 50
elif params.data_dir in ['BePKT']:
    NUM_QUESTIONS = 553

# log 文件名加 _true5fold, 避免覆盖旧协议的结果
save_log = './log'
if not os.path.exists(save_log):
    os.makedirs(save_log)
log_file = os.path.join(
    save_log + '/wo_ReasonLoss{}_d{}_{}_{}_true5fold_test_result.txt'.format(
        model_tag, params.emb_size, params.data_dir, params.kc_mode))  # [fix] 加 kc_mode, 以防 skills 跑完覆盖 kcmask 的结果
log = open(log_file, 'w')
print(f'运行配置: model={model_tag} data={params.data_dir} kc_mode={params.kc_mode} '
      f'NUM_SKILLS={NUM_SKILLS} emb={params.emb_size} seed={params.seed}', file=log, flush=True)

rmse_test_list = []
mae_test_list = []
acc_test_list = []
auc_test_list = []
now = time.time()

folds_to_run = [params.fold] if params.fold >= 0 else range(5)

for fold_idx in folds_to_run:
    print(f'================ Fold {fold_idx} ================')
    print(f'================ Fold {fold_idx} ================', file=log, flush=True)

    # [7] 每折独立种子 (学生级真5折: 每折是不同的学生划分, 也避免RNG只跑同一划分)
    seed = params.seed + fold_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # [7] 每折读对应数据划分: 数据集创建移入 fold 循环内
    params.fold = fold_idx
    train_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Train')
    valid_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Valid')
    test_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Test')

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=10, verbose=True)

    # [fix] ckpt 目录加 kc_mode 后缀, 否则 skills 跑完会覆盖 kcmask 的 kt_model_best.pt
    _ckpt_tag = '' if params.kc_mode == 'kcmask' else '_' + params.kc_mode
    save_model_file = os.path.join(
        './fold{}/{}/dim{}S31R3-True{}'.format(fold_idx, params.data_dir, params.emb_size, _ckpt_tag))
    if not os.path.exists(save_model_file):
        os.makedirs(save_model_file)

    model = StudentKnowledgeModel(NUM_QUESTIONS, NUM_SKILLS, params.emb_size, 8, params.device,
                                  use_debug=USE_DEBUG).to(params.device)
    model_adam = optim.Adam(model.parameters(), lr=params.lr)
    loss_func = lossFunc(params.device)

    if getattr(params, 'nan_debug', False):
        print('[nan_debug] autograd.set_detect_anomaly(True) 已开启 ' 
              '(用于定位 NaN/Inf 来源, 训练会变慢)')
        torch.autograd.set_detect_anomaly(True)

    for epoch in range(params.max_iter):
        print(f'current epoch is {epoch+1}')
        model.train()
        for batch in tqdm.tqdm(train_dataloader, desc='Training'):
            # gen_data 返回 9 项 (末两项为代码张量, 本模型不用)
            problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor, debug_tensor, _cf, _ct = batch

            pred_score, pred_reason_logic, pred_reason_concept, logic_weight, concept_weight = model(
                problem_tensor, skills_tensor, label_tensor, reason_tensor, time_tensor[:, 1:],
                global_debug=debug_tensor)
            pred_loss = loss_func(pred_score[:, 1:], label_tensor[:, 1:], note_tensor[:, 1:])

            # ==== NaN 检查: 跳过异常 batch ====
            if torch.isnan(pred_loss) or torch.isinf(pred_loss):
                print(f'  ⚠️ 跳过batch: loss={pred_loss.item()}')
                model_adam.zero_grad()
                continue
            # ======================================

            total_loss = pred_loss
            model_adam.zero_grad()
            total_loss.backward()
            # [fix] 梯度含 NaN/Inf 时跳过该 batch: 否则 clip_grad_norm 的缩放系数会变成 NaN,
            #       把全部参数污染成 NaN → 下一步 BCE 收到 NaN → CUDA device-side assert
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(gn):
                print(f'  [!] 跳过batch(梯度非有限): grad_norm={gn.item()}')
                model_adam.zero_grad()
                continue
            model_adam.step()

        model.eval()
        all_preds = []
        all_labels = []
        loss_list = torch.Tensor([]).to(params.device)
        with torch.no_grad():
            for batch in tqdm.tqdm(valid_dataloader, desc='Validating'):
                problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor, debug_tensor, _cf, _ct = batch
                pred, _, _, _, _ = model(
                    problem_tensor, skills_tensor, label_tensor, reason_tensor, time_tensor[:, 1:],
                    global_debug=debug_tensor)
                all_preds.append(pred[:, 1:])
                all_labels.append(note_tensor[:, 1:])
                loss = loss_func(pred[:, 1:], label_tensor[:, 1:], note_tensor[:, 1:])
                loss_list = torch.cat((loss_list, torch.tensor([loss], device=params.device)), dim=0)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        valid_rmse, valid_mae, valid_acc, valid_auc = performance(all_labels, all_preds)
        print(f'valid_rmse: {valid_rmse:.4f}  valid_mae: {valid_mae:.4f}  valid_acc: {valid_acc:.4f}  valid_auc: {valid_auc:.4f}')

        early_stopping(loss_list.mean(), model,
                       save_path=os.path.join(save_model_file, 'kt_model_best.pt'))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    load_model_path = os.path.join(save_model_file, 'kt_model_best.pt')
    model.load_state_dict(torch.load(load_model_path, weights_only=True))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, desc='Testing'):
            problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor, debug_tensor, _cf, _ct = batch
            pred, _, _, _, _ = model(
                problem_tensor, skills_tensor, label_tensor, reason_tensor, time_tensor[:, 1:],
                global_debug=debug_tensor)
            all_preds.append(pred[:, 1:])
            all_labels.append(note_tensor[:, 1:])
            loss = loss_func(pred[:, 1:], label_tensor[:, 1:], note_tensor[:, 1:])

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        rmse, mae, acc, auc = performance(all_labels, all_preds)
        rmse_test_list.append(rmse)
        mae_test_list.append(mae)
        acc_test_list.append(acc)
        auc_test_list.append(auc)
        print(f'[Fold {fold_idx}] test_rmse: {rmse:.4f}  test_mae: {mae:.4f}  test_acc: {acc:.4f}  test_auc: {auc:.4f}')
        print(f'[Fold {fold_idx}] test_rmse: {rmse:.4f}  test_mae: {mae:.4f}  test_acc: {acc:.4f}  test_auc: {auc:.4f}', file=log, flush=True)

print('average test rmse:', np.round(np.mean(rmse_test_list), decimals=4), u'\u00B1',
      np.round(np.std(rmse_test_list), decimals=4))
print('average test rmse:', np.round(np.mean(rmse_test_list), decimals=4), u'\u00B1',
      np.round(np.std(rmse_test_list), decimals=4), file=log)
print('average test mae:', np.round(np.mean(mae_test_list), decimals=4), u'\u00B1',
      np.round(np.std(mae_test_list), decimals=4))
print('average test mae:', np.round(np.mean(mae_test_list), decimals=4), u'\u00B1',
      np.round(np.std(mae_test_list), decimals=4), file=log)
print('average test acc:', np.round(np.mean(acc_test_list), decimals=4), u'\u00B1',
      np.round(np.std(acc_test_list), decimals=4))
print('average test acc:', np.round(np.mean(acc_test_list), decimals=4), u'\u00B1',
      np.round(np.std(acc_test_list), decimals=4), file=log)
print('average test auc:', np.round(np.mean(auc_test_list), decimals=4), u'\u00B1',
      np.round(np.std(auc_test_list), decimals=4))
print('average test auc:', np.round(np.mean(auc_test_list), decimals=4), u'\u00B1',
      np.round(np.std(auc_test_list), decimals=4), file=log)

end = time.time()
print('total running time:{} min'.format((end - now) / 60))
print('total running time:{} min'.format((end - now) / 60), file=log)
log.close()
print('此为 mlkt4 (true 5-fold) baseline')
