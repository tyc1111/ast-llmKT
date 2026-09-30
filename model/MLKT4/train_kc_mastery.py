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
import hashlib
import re

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
parser.add_argument('--data_subdir', type=str, default=None,
                    help='数据子目录; dualkc 默认 MSDGKT_perkc_dualkc, 旧模式默认 MSDGKT')

# model params
parser.add_argument('--device', type=str, default=device, help='device')
parser.add_argument('--emb_size', type=int, default=64, help='embedding size (default 64 for scalar mastery)')

# 实验参数
parser.add_argument('--fold', type=int, default=-1, help='只跑单个fold(-1=全部5折), 用于快速实验试水')
parser.add_argument('--run_tag', type=str, default='', help='独立实验输出标签（字母、数字、下划线或短横线）')
parser.add_argument('--kc_mode', type=str, default='kcmask', choices=['kcmask', 'skills', 'dualkc'],
                    help='知识组件来源: kcmask=旧KC, skills=人工Skills, dualkc=fold专属 C/M-KC')
parser.add_argument('--mode_ctx', action='store_true',
                    help='路线② 模式识别器: 因果窗口(W=5)可学习识别器, 连续学习概率p; 默认off=原EMA硬门(0.8368基线)')
parser.add_argument('--code_assets', action='store_true',
                    help='构建 per-step 代码张量(code_feat 35维 / code_tokens 128); 默认关(回滚后模型不用, 关掉省时省内存)')
parser.add_argument('--no_graph', action='store_true',
                    help='不用KC图(保持单位阵=全连接); 用于 kc_mode=skills 等与 ast_kc_mapping 尺寸不匹配的情形')

params = parser.parse_args()
if params.run_tag and not re.fullmatch(r'[A-Za-z0-9_-]+', params.run_tag):
    parser.error('--run_tag must contain only letters, digits, underscores or hyphens')
if params.data_subdir is None:
    params.data_subdir = 'MSDGKT_perkc_dualkc' if params.kc_mode == 'dualkc' else 'MSDGKT'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if params.kc_mode == 'dualkc':
    dual_data_manifest = os.path.join(
        BASE_DIR, '..', '..', 'data', params.data_dir, params.data_subdir,
        'fivefold_export_manifest.json'
    )
    if not os.path.isfile(dual_data_manifest):
        raise FileNotFoundError(
            f'Dual-KC Per-KC data not found: {dual_data_manifest}; '
            'run export_perkc_dualkc_5fold.py first.'
        )
    with open(dual_data_manifest, 'r', encoding='utf-8') as f:
        dual_data_info = json.load(f)
    if int(dual_data_info.get('max_sequence_length', -1)) != params.max_length:
        raise ValueError(
            f"Dual-KC data sequence length is {dual_data_info.get('max_sequence_length')}, "
            f"but --max_length={params.max_length}; export data with matching --seq-len."
        )

# ==== 动态读取知识组件数量 (kc_mode 决定来源) ====
KC_MAPPING_PATH = os.path.join(BASE_DIR, '..', '..', 'data', params.data_dir, 'ast_kc_mapping.json')
DUAL_MAPPING_TEMPLATE = os.path.join(
    BASE_DIR, '..', '..', 'data', 'BePKT', 'dual_kc_taskA_final',
    'fold{}', 'post_order_fix', 'kt_mapping.json'
)
if params.kc_mode == 'dualkc':
    with open(DUAL_MAPPING_TEMPLATE.format(0), 'r', encoding='utf-8') as f:
        dual_mapping = json.load(f)
    NUM_SKILLS = int(dual_mapping['num_kcs'])
    if int(dual_mapping['concept_kcs']) + int(dual_mapping['malformed_kcs']) != NUM_SKILLS:
        raise ValueError('Fold-0 Dual-KC branch counts do not sum to num_kcs')
    del dual_mapping
    print(f'[Per-KC Mastery] 使用 fold 专属 Dual-KC 词表: {NUM_SKILLS} (C+M)')
elif params.kc_mode == 'kcmask':
    if os.path.exists(KC_MAPPING_PATH):
        with open(KC_MAPPING_PATH, 'r', encoding='utf-8') as f:
            kc_mapping = json.load(f)
        NUM_SKILLS = kc_mapping['num_kcs']
        print(f'[Per-KC Mastery] 使用 {NUM_SKILLS} 个自动KCs (kc_mode=kcmask)')
    else:
        raise FileNotFoundError(f'需要KC映射文件: {KC_MAPPING_PATH}')
else:
    # skills 模式: 从数据的 Skills 列推断人工skill数量 (最大索引+1)
    # [fix] 原来读非折目录 data/<ds>/MSDGKT/q_a_train_detailed.csv, 但 BePKT 只有 fold0..4 → FileNotFoundError
    _base = os.path.join(BASE_DIR, '..', '..', 'data', params.data_dir, params.model)
    _paths = [os.path.join(_base, 'fold0', f'q_a_{s}_detailed.csv') for s in ['train', 'valid', 'test']]
    if not all(os.path.exists(p) for p in _paths):
        _paths = [os.path.join(_base, f'q_a_{s}_detailed.csv') for s in ['train', 'valid', 'test']]
    max_skill = -1
    for _p in _paths:
        if os.path.exists(_p):
            for _s in pd.read_csv(_p, usecols=['Skills'])['Skills']:
                _v = eval(_s) if isinstance(_s, str) else _s
                if len(_v):
                    max_skill = max(max_skill, max(_v))
    if max_skill < 0:
        raise FileNotFoundError(f'无法从数据推断 Skills 数(检查路径): {_paths}')
    NUM_SKILLS = max_skill + 1
    print(f'[Per-KC Mastery] 使用 {NUM_SKILLS} 个人工Skills (kc_mode=skills, max id={max_skill})')

if params.data_dir in ['Code-S', 'Code-F']:
    NUM_QUESTIONS = 50
elif params.data_dir in ['BePKT']:
    NUM_QUESTIONS = 553


def forward_from_batch(model, batch):
    problem_tensor, kc_tensor, label_tensor, _note_tensor, reason_tensor, time_tensor = batch[:6]
    debug_tensor = batch[6]
    if params.kc_mode == 'dualkc':
        query_mask, event_mask = batch[9], batch[10]
        return model(problem_tensor, kc_tensor, label_tensor, reason_tensor, time_tensor[:, 1:],
                     global_debug=debug_tensor, query_kc_mask=query_mask, update_kc_mask=event_mask)
    return model(problem_tensor, kc_tensor, label_tensor, reason_tensor, time_tensor[:, 1:],
                 global_debug=debug_tensor)

# 数据已有TimeInterval列 (由data_save生成), 无需再调用mlkt_utils
# [7] 数据集/DataLoader 移到每折内创建, 以便读取 fold{fold_idx}/ 数据划分 (真5折)

# Log file
save_log = './log'
if not os.path.exists(save_log):
    os.makedirs(save_log)
# [A/B] 实验标签: 默认(base)文件名不变; 增强版加后缀区分
_ab_tag = ''
if params.mode_ctx:
    _ab_tag += '_mc'
if params.kc_mode == 'skills':
    _ab_tag += '_sk'      # [fix] 防 skills 运行覆盖 kcmask 基线的日志/ckpt (基线 tag 仍为空)
if params.kc_mode == 'dualkc':
    _ab_tag += '_dualkc'
if params.no_graph:
    _ab_tag += '_ng'
if 'fulltrain' in params.data_subdir:
    _ab_tag += '_fulltrain'
if params.run_tag:
    _ab_tag += '_' + params.run_tag
log_file = os.path.join(
    save_log + f'/KCMastery_d{params.emb_size}_{params.data_dir}{_ab_tag}_test_result.txt'
)
log = open(log_file, 'w')

rmse_test_list = []
mae_test_list = []
acc_test_list = []
auc_test_list = []
now = time.time()

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

result_file = os.path.splitext(log_file)[0] + '_run.json'
run_record = {
    'status': 'running', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'argv': sys.argv, 'configuration': {k: str(v) if k == 'device' else v for k, v in vars(params).items()},
    'effective_seed_base': seed, 'seed_note': 'Historical effective seed retained for paired comparison; --seed is not applied.',
    'experiment_protocol': 'P0 Full-Train KCQuery / Legacy-200 diagnostic' if 'fulltrain' in params.data_subdir else 'legacy',
    'data_manifest_sha256': file_sha256(dual_data_manifest) if params.kc_mode == 'dualkc' else None,
    'code_sha256': {name: file_sha256(os.path.join(BASE_DIR, name)) for name in
                    ['train_kc_mastery.py', 'mlkt4_kc_mastery.py', 'gen_data.py', 'eval_mlkt.py']},
    'torch_version': torch.__version__,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
    'fold_results': [],
}

def write_run_record():
    temporary = result_file + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(run_record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, result_file)

write_run_record()
print('configuration: ' + json.dumps(run_record['configuration'], ensure_ascii=False), file=log, flush=True)
print('protocol: ' + run_record['experiment_protocol'], file=log, flush=True)

folds_to_run = [params.fold] if params.fold >= 0 else range(5)
print(f'运行折数: {list(folds_to_run)}')
# [log] 折信息/实验标签落盘: 避免逐折结果只在 stdout 而拿错折或丢失
print(f'运行折数: {list(folds_to_run)} | tag={_ab_tag or "(off)"} | kc_mode={params.kc_mode} | code_assets={params.code_assets} | no_graph={params.no_graph}',
      file=log, flush=True)
for fold_idx in folds_to_run:
    print(f'\n{"="*60}')
    print(f'Fold {fold_idx}')
    print(f'{"="*60}')
    print(f'\n{"="*60}', file=log, flush=True)
    print(f'Fold {fold_idx}', file=log, flush=True)
    print(f'{"="*60}', file=log, flush=True)
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
    params.num_kcs = NUM_SKILLS
    params.data_subdir = params.data_subdir
    train_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Train')
    valid_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Valid')
    test_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Test')
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    # ---- 创建 VAE 模型 (单步KL) ----
    model = VAEPerKCMasteryModel(
        NUM_QUESTIONS, NUM_SKILLS, params.emb_size, K=8,
        device=params.device, beta=0.001,
        mode_ctx=params.mode_ctx
    ).to(params.device)

    # 加载KC邻接矩阵 (仅 kc_mode 与图尺寸匹配时; --no_graph 或尺寸不符则保持单位阵=全连接)
    if params.no_graph:
        print(f"  [graph] --no_graph: 跳过KC图, 使用单位阵(全连接, {NUM_SKILLS}节点)")
        print(f"  [graph] --no_graph: 跳过KC图, 使用单位阵(全连接, {NUM_SKILLS}节点)", file=log, flush=True)
    else:
        fold_mapping_path = (DUAL_MAPPING_TEMPLATE.format(fold_idx)
                             if params.kc_mode == 'dualkc' else KC_MAPPING_PATH)
        if os.path.exists(fold_mapping_path):
            with open(fold_mapping_path, 'r', encoding='utf-8') as f:
                kc_data = json.load(f)
            if int(kc_data.get('num_kcs', NUM_SKILLS)) != NUM_SKILLS:
                raise ValueError(f'Fold {fold_idx} mapping KC size differs from model size')
            if 'kc_adjacency' in kc_data:
                adj = torch.tensor(kc_data['kc_adjacency'], dtype=torch.float, device=params.device)
                if adj.shape[0] != NUM_SKILLS:
                    msg = (f"  [graph] ⚠️ KC图尺寸 {tuple(adj.shape)} != NUM_SKILLS {NUM_SKILLS} "
                           f"(kc_mode={params.kc_mode}) → 跳过该图, 使用单位阵(全连接)")
                    print(msg)
                    print(msg, file=log, flush=True)
                else:
                    model.kc_adj = adj
                    edge_count = int(adj.sum().item() - NUM_SKILLS)
                    print(f"  加载KC邻接矩阵: {NUM_SKILLS}节点, {edge_count}条边")
                    print(f"  加载KC邻接矩阵: {NUM_SKILLS}节点, {edge_count}条边 (源: {os.path.basename(fold_mapping_path)})",
                          file=log, flush=True)
            elif params.kc_mode == 'dualkc':
                raise ValueError(f'Fold {fold_idx} Dual-KC mapping has no adjacency matrix')
        elif params.kc_mode == 'dualkc':
            raise FileNotFoundError(f'Fold-specific Dual-KC mapping not found: {fold_mapping_path}')

    model_adam = optim.Adam(model.parameters(), lr=params.lr)

    # ---- 训练 ----
    for epoch in range(params.max_iter):
        print(f'epoch {epoch+1}')
        model.set_epoch(epoch)  # 更新KL退火
        model.train()
        for batch in tqdm.tqdm(train_dataloader, desc='Training'):
            note_tensor = batch[3]
            label_tensor = batch[2]
            pred_score, kl_step, extra = forward_from_batch(model, batch)
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
                label_tensor = batch[2]
                note_tensor = batch[3]
                pred, kl_step, extra = forward_from_batch(model, batch)
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
        print(f'  valid_rmse: {valid_rmse:.4f}  valid_mae: {valid_mae:.4f}  valid_acc: {valid_acc:.4f}  valid_auc: {valid_auc:.4f}',
              file=log, flush=True)
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
            label_tensor = batch[2]
            note_tensor = batch[3]
            pred, _, _ = forward_from_batch(model, batch)
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
        print(f'  [fold {fold_idx}] test_rmse: {rmse:.4f}  test_mae: {mae:.4f}  test_acc: {acc:.4f}  test_auc: {auc:.4f}',
              file=log, flush=True)
        run_record['fold_results'].append({
            'fold': fold_idx, 'effective_seed': seed + fold_idx,
            'rmse': float(rmse), 'mae': float(mae), 'acc': float(acc), 'auc': float(auc),
            'epochs_run': epoch + 1, 'early_stopped': bool(early_stopping.early_stop),
            'checkpoint': os.path.abspath(load_model_path),
        })
        write_run_record()

# ---- 最终结果 ----
print(f'\n{"="*60}')
print(f'Per-KC Mastery 最终结果 (emb_size={params.emb_size}, {params.data_dir})')
print(f'{"="*60}')
print(f'RMSE: {np.mean(rmse_test_list):.4f} ± {np.std(rmse_test_list):.4f}')
print(f'MAE:  {np.mean(mae_test_list):.4f} ± {np.std(mae_test_list):.4f}')
print(f'ACC:  {np.mean(acc_test_list):.4f} ± {np.std(acc_test_list):.4f}')
print(f'AUC:  {np.mean(auc_test_list):.4f} ± {np.std(auc_test_list):.4f}')
print(f'总耗时: {(time.time()-now)/60:.2f} min')
print(f'总耗时: {(time.time()-now)/60:.2f} min', file=log, flush=True)
print('逐折AUC: ' + ' / '.join(f'{v:.4f}' for v in auc_test_list), file=log, flush=True)

print(f'\n{"="*60}', file=log)
print(f'Per-KC Mastery 最终结果 (emb_size={params.emb_size}, {params.data_dir})', file=log)
print(f'{"="*60}', file=log)
print(f'RMSE: {np.mean(rmse_test_list):.4f} ± {np.std(rmse_test_list):.4f}', file=log)
print(f'MAE:  {np.mean(mae_test_list):.4f} ± {np.std(mae_test_list):.4f}', file=log)
print(f'ACC:  {np.mean(acc_test_list):.4f} ± {np.std(acc_test_list):.4f}', file=log)
print(f'AUC:  {np.mean(auc_test_list):.4f} ± {np.std(auc_test_list):.4f}', file=log)
log.close()
run_record.update(status='completed', elapsed_minutes=(time.time() - now) / 60,
                  completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
run_record['aggregate'] = {
    name: {'mean': float(np.mean(values)), 'std': float(np.std(values))}
    for name, values in [('rmse', rmse_test_list), ('mae', mae_test_list), ('acc', acc_test_list), ('auc', auc_test_list)]
}
write_run_record()
