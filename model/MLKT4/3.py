"""
ast_kc_discovery.py
=====================
从学生代码的AST中自动发现知识组件(KCs)

用法:
  python ast_kc_discovery.py [dataset]
  
  dataset: Code-S (默认) 或 BePKT

流程:
  1. 读取CodeStates中的所有代码 → 解析AST → 提取子树
  2. 构建Transformer编码子树 → Softmax重要性打分
  3. 按重要性加权聚类 → 形成KCs
  4. 输出每道题的KC_mask映射
"""

import json
import os
import sys
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 使用PyTorch实现KMeans,避免sklearn的Windows兼容问题

# ============================================================
# 1. AST解析工具
# ============================================================

def init_java_parser():
    """初始化Java AST解析器"""
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_java as tsjava
        JAVA_LANGUAGE = Language(tsjava.language())
        parser = Parser(JAVA_LANGUAGE)
        return parser
    except ImportError:
        print("请安装: pip install tree-sitter tree-sitter-java")
        sys.exit(1)


def init_c_parser():
    """初始化C AST解析器 (用于BePKT数据集)"""
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_c as tsc
        C_LANGUAGE = Language(tsc.language())
        parser = Parser(C_LANGUAGE)
        return parser
    except ImportError:
        print("请安装: pip install tree-sitter-c")
        sys.exit(1)


def parse_code_to_ast(parser, code_text):
    """将代码文本解析为AST, 返回根节点"""
    try:
        tree = parser.parse(code_text.encode())
        return tree.root_node
    except Exception as e:
        return None


# ============================================================
# 2. 子树提取
# ============================================================

# 有知识区分度的AST节点类型
# 排除所有方法都有的通用结构(方法定义/代码块/变量声明/return)
# 只保留能区分不同知识点的结构
INTERESTING_TYPES = {
    'for_statement', 'while_statement', 'do_statement',
    'if_statement', 'else_clause', 'switch_statement',
    'method_invocation', 'throw_statement',
    'assignment_expression',
    'binary_expression', 'unary_expression',
    'array_creation', 'object_creation',
    'try_statement', 'catch_clause',
    'conditional_expression',
    # C/C++特有结构
    'call_expression', 'pointer_expression',
    'sizeof_expression', 'cast_expression',
    'struct_specifier', 'enum_specifier',
    'preproc_include', 'preproc_function_def',
    'string_literal', 'number_literal',
}


def extract_subtrees(node, max_depth=3, current_depth=0):
    """从AST节点中提取所有有意义的子树
    
    返回: [(子树类型, [子节点类型列表], 深度, 特征), ...]
    新增结构指纹: 深度、子节点数、嵌套循环标记、操作符类型、变量数
    """
    results = []
    if current_depth > max_depth:
        return results

    node_type = node.type
    
    # 只保留有意义的语法结构
    if node_type in INTERESTING_TYPES:
        # 子节点类型 + 结构指纹
        child_types = []
        has_nested_loop = False
        binary_op = 'none'
        var_ref_count = 0
        has_method_call = False
        
        for child in node.children:
            t = child.type
            if t not in {'{', '}', '(', ')', ';', ','}:
                child_types.append(t)
            
            # 检测嵌套循环
            if t in {'for_statement', 'while_statement', 'do_statement'}:
                has_nested_loop = True
            
            # 统计变量引用
            if t == 'identifier':
                var_ref_count += 1
            
            # 提取二元操作符
            if t in {'>', '<', '>=', '<=', '==', '!=', '+=', '-=',
                     '+', '-', '*', '/', '%', '&&', '||', '!', '='}:
                binary_op = t
            
            # 方法调用检测
            if t in ('method_invocation', 'argument_list'):
                has_method_call = True
        
        # 构建特征序列 (包含结构指纹)
        features = [node_type]
        features.extend(child_types[:5])
        features.append(f'd{current_depth}')
        features.append(f'c{len(child_types)}')
        if has_nested_loop:
            features.append('nl')
        if binary_op != 'none':
            features.append(f'op_{binary_op}')
        if var_ref_count > 0:
            features.append(f'v{var_ref_count}')
        if has_method_call:
            features.append('mc')
        
        results.append({
            'type': node_type,
            'children': child_types[:8],
            'depth': current_depth,
            'start_byte': node.start_byte,
            'end_byte': node.end_byte,
            'features': features,  # 结构化特征
        })
    
    # 递归子节点
    for child in node.children:
        results.extend(extract_subtrees(child, max_depth, current_depth + 1))
    
    return results
    
    # 递归子节点
    for child in node.children:
        results.extend(extract_subtrees(child, max_depth, current_depth + 1))
    
    return results


def flatten_subtree(subtree):
    """将子树展平为类型序列, 用于编码"""
    if 'features' in subtree:
        return subtree['features']
    seq = [subtree['type']]
    seq.extend(subtree['children'])
    return seq


# ============================================================
# 3. 构建AST类型词表
# ============================================================

def build_type_vocab(all_subtrees):
    """从所有子树中构建AST节点类型词表"""
    all_types = set()
    for sub in all_subtrees:
        all_types.add(sub['type'])
        all_types.update(sub['children'])
    
    type_vocab = {t: i+1 for i, t in enumerate(sorted(all_types))}
    type_vocab['<PAD>'] = 0
    return type_vocab


def subtree_to_ids(subtree, type_vocab, max_len=16):
    """将子树编码为类型ID序列"""
    seq = flatten_subtree(subtree)
    ids = [type_vocab.get(t, 0) for t in seq[:max_len]]
    # padding
    ids += [0] * (max_len - len(ids))
    return ids


# ============================================================
# 4. Transformer重要性打分器
# ============================================================

class ImportanceScorer(nn.Module):
    """
    给每道题的代码中提取的子树打分
    
    输入: 一道题的所有代码提交 → 所有子树
    输出: 每个子树的重要性权重 (Softmax归一化)
    
    核心思想:
      如果某类子树经常出现在"高分数"代码中
      → 它是重要的KC
      如果某类子树在"低分数"中也常见
      → 它可能是学生容易出错的KC
    """
    
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=500):
        super().__init__()
        self.device_prop = None
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, subtree_ids, mask):
        """
        Args:
            subtree_ids: [B, L]  B=代码数量, L=子树序列长度(padded)
            mask: [B, L]  True=padding位置
        Returns:
            importance: [B, L]  Softmax归一化的重要性权重
        """
        B, L = subtree_ids.shape
        
        # 1. 嵌入
        x = self.embedding(subtree_ids)  # [B, L, D]
        x = x + self.pos_encoding[:, :L, :]
        
        # 2. Transformer编码 (子树间的关系)
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, L, D]
        
        # 3. 打分 + Softmax
        scores = self.score_head(x).squeeze(-1)  # [B, L]
        # 将padding位置的分数设为负无穷
        scores = scores.masked_fill(mask, float('-inf'))
        importance = F.softmax(scores, dim=-1)  # [B, L], 每行和为1
        importance = importance.masked_fill(mask, 0.0)
        
        return importance  # [B, L]

    @property
    def device(self):
        if self.device_prop is None:
            self.device_prop = next(self.parameters()).device
        return self.device_prop


# ============================================================
# 5. 对比学习训练
# ============================================================

class CodePairDataset(Dataset):
    """同一道题的"高分代码vs低分代码"对"""
    
    def __init__(self, problem_groups, type_vocab, max_len=16):
        self.samples = []
        
        for pid, group in problem_groups.items():
            if len(group) < 2:
                continue
            
            # 高分代码 (score >= 0.8) 和 低分代码 (score < 0.5)
            high = [(s, c) for s, c, _ in group if s >= 0.8]
            low = [(s, c) for s, c, _ in group if s < 0.5]
            
            if not high or not low:
                continue
            
            # 构建对比对
            for h_score, h_subtrees in high[:3]:
                for l_score, l_subtrees in low[:3]:
                    self.samples.append({
                        'high_subtrees': [subtree_to_ids(s, type_vocab, max_len) for s in h_subtrees],
                        'low_subtrees': [subtree_to_ids(s, type_vocab, max_len) for s in l_subtrees],
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_subtrees(batch, max_subtrees=30):
    """批量处理子树序列"""
    high_batch, low_batch = [], []
    high_mask_batch, low_mask_batch = [], []
    
    for item in batch:
        # 处理high
        h = item['high_subtrees'][:max_subtrees]
        seq_len = len(h)
        h_padded = h + [[0]*16] * (max_subtrees - seq_len)
        h_mask = [False]*seq_len + [True]*(max_subtrees - seq_len)
        high_batch.append(h_padded)
        high_mask_batch.append(h_mask)
        
        # 处理low
        l = item['low_subtrees'][:max_subtrees]
        seq_len = len(l)
        l_padded = l + [[0]*16] * (max_subtrees - seq_len)
        l_mask = [False]*seq_len + [True]*(max_subtrees - seq_len)
        low_batch.append(l_padded)
        low_mask_batch.append(l_mask)
    
    return {
        'high': torch.tensor(high_batch),    # [B, L, 10]
        'high_mask': torch.tensor(high_mask_batch),  # [B, L]
        'low': torch.tensor(low_batch),
        'low_mask': torch.tensor(low_mask_batch),
    }


def train_importance_scorer(model, dataloader, epochs=20, lr=1e-3):
    """
    训练目标:
      高分代码中的"关键子树"应该获得高重要性
      让模型学会"哪些子树模式能预测高分"
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            high = batch['high'].to(model.device)      # [B, L, 10]
            high_mask = batch['high_mask'].to(model.device)  # [B, L]
            low = batch['low'].to(model.device)
            low_mask = batch['low_mask'].to(model.device)
            
            # 展平最后一位 (subtree token序列)
            B, L, S = high.shape
            high_flat = high.view(B, L * S)
            low_flat = low.view(B, L * S)
            
            # 重建mask
            high_mask_flat = high_mask.repeat_interleave(S, dim=1)
            low_mask_flat = low_mask.repeat_interleave(S, dim=1)
            
            # 高分代码的子树应该获得更集中的注意力
            high_imp = model(high_flat, high_mask_flat)
            low_imp = model(low_flat, low_mask_flat)
            
            # 损失: 高分代码的子树重要性分布应该更集中 (熵更小)
            # 低分代码的分布更分散 (熵更大)
            high_entropy = -(high_imp * torch.log(high_imp + 1e-8)).sum(dim=1).mean()
            low_entropy = -(low_imp * torch.log(low_imp + 1e-8)).sum(dim=1).mean()
            
            # 训练目标: 高分代码的子树注意力更集中
            loss = high_entropy - low_entropy + 0.5 * low_entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


# ============================================================
# 6. 加权聚类 → KCs
# ============================================================

def cluster_into_kcs(subtree_vectors, importance_weights, n_clusters=35, max_iter=50):
    """
    按重要性加权的KMeans聚类 (向量化numpy实现)
    """
    X = (subtree_vectors * importance_weights.unsqueeze(1)).detach().numpy().astype(np.float32)
    N, D = X.shape
    
    # 用随机采样初始化中心 (更快)
    rng = np.random.RandomState(42)
    idx = rng.choice(N, n_clusters, replace=False)
    centers = X[idx].copy()
    
    for _ in range(max_iter):
        # 距离矩阵: [n_clusters, N] (向量化!)
        dists = np.zeros((n_clusters, N), dtype=np.float32)
        for k in range(n_clusters):
            diff = X - centers[k:k+1]
            dists[k] = np.sum(diff * diff, axis=1)  # 平方距离
        
        labels = np.argmin(dists, axis=0)
        
        # 更新中心
        new_centers = centers.copy()
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centers[k] = X[mask].mean(axis=0)
        
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return labels, centers


def analyze_kc_patterns(kc_labels, all_subtrees, n_clusters):
    """分析每个KC包含的代码模式, 生成可解释的描述"""
    kc_patterns = {}
    
    for kc_id in range(n_clusters):
        members = [i for i, l in enumerate(kc_labels) if l == kc_id]
        if not members:
            continue
        
        # 统计该KC中最常见的子树类型
        type_counter = Counter()
        for idx in members:
            sub = all_subtrees[idx]
            type_counter[sub['type']] += 1
        
        top_type = type_counter.most_common(1)[0][0]
        
        # 生成描述
        type_descriptions = {
            'for_statement': 'for循环',
            'while_statement': 'while循环', 
            'do_statement': 'do-while循环',
            'if_statement': 'if条件分支',
            'else_clause': 'else分支',
            'switch_statement': 'switch分支',
            'method_invocation': '方法调用',
            'call_expression': '函数调用',
            'throw_statement': 'throw语句',
            'binary_expression': '二元运算',
            'unary_expression': '一元运算',
            'assignment_expression': '赋值操作',
            'array_creation': '数组创建',
            'object_creation': '对象创建',
            'try_statement': 'try异常处理',
            'catch_clause': 'catch子句',
            'conditional_expression': '三元表达式',
            'pointer_expression': '指针运算',
            'sizeof_expression': 'sizeof运算',
            'cast_expression': '类型转换',
            'struct_specifier': '结构体',
            'enum_specifier': '枚举',
            'preproc_include': '头文件引入',
            'preproc_function_def': '宏定义',
            'string_literal': '字符串常量',
            'number_literal': '数字常量',
        }
        description = type_descriptions.get(top_type, top_type)
        
        # 如果该KC有多种类型混合
        if len(type_counter) >= 3:
            all_types = [type_descriptions.get(t, t) for t, _ in type_counter.most_common(3)]
            description = '+'.join(all_types)
        
        # ==== 提取结构指纹特征 ====
        feature_fingerprint = {}
        op_counter = Counter()
        depth_counter = Counter()
        child_count_counter = Counter()
        nested_loop_count = 0
        method_call_count = 0
        for idx in members:
            sub = all_subtrees[idx]
            if 'features' in sub:
                for feat in sub['features']:
                    if feat.startswith('op_'):
                        op_counter[feat] += 1
                    elif feat.startswith('d'):
                        depth_counter[feat] += 1
                    elif feat.startswith('c'):
                        child_count_counter[feat] += 1
                    elif feat == 'nl':
                        nested_loop_count += 1
                    elif feat == 'mc':
                        method_call_count += 1
        if op_counter:
            feature_fingerprint['operators'] = [k for k, _ in op_counter.most_common(3)]
        if depth_counter:
            feature_fingerprint['depths'] = [k for k, _ in depth_counter.most_common(3)]
        if child_count_counter:
            feature_fingerprint['child_counts'] = [k for k, _ in child_count_counter.most_common(3)]
        feature_fingerprint['nested_loop_ratio'] = round(nested_loop_count / max(len(members), 1), 2)
        feature_fingerprint['method_call_ratio'] = round(method_call_count / max(len(members), 1), 2)
        # ================================
        
        kc_patterns[kc_id] = {
            'num_members': len(members),
            'top_type': top_type,
            'description': description,
            'type_distribution': dict(type_counter.most_common(5)),
            'avg_depth': np.mean([all_subtrees[i]['depth'] for i in members]),
            'feature_fingerprint': feature_fingerprint,
        }
    
    return kc_patterns


# ============================================================
# 7. 主流程
# ============================================================

# ============================================================
# 辅助: 题目文本→特征向量
# ============================================================

def text_to_vec(text, dim=128, stop_words=None):
    """基于字符3-gram哈希的文本向量, 无需外部依赖"""
    if stop_words is None:
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                      'to', 'of', 'in', 'for', 'on', 'at', 'by',
                      'and', 'or', 'but', 'with', 'from', 'this'}
    vec = np.zeros(dim, dtype=np.float32)
    text = re.sub(r'[^a-z0-9]', ' ', text.lower())
    # 词级特征
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    for w in words:
        idx = hash(w) % dim
        vec[idx] += 1.0
    # 字符3-gram特征
    clean = text.replace(' ', '')
    for i in range(len(clean) - 2):
        gram = clean[i:i+3]
        idx = hash(gram) % dim
        vec[idx] += 0.5
    # 归一化
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def load_problem_texts(data_dir):
    """加载题目Requirement并转为文本向量"""
    prob_path = os.path.join(data_dir, 'problem_skills.csv')
    if not os.path.exists(prob_path):
        prob_path = os.path.join(data_dir, 'problem.csv')
    problem_texts = {}
    if os.path.exists(prob_path):
        import csv
        with open(prob_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                pid = row.get('ProblemID', '')
                # 优先用Requirement, 其次用Title
                text = row.get('Requirement', '') or row.get('Title', '') or ''
                problem_texts[pid] = text_to_vec(text)
        print(f"  加载 {len(problem_texts)} 道题的文本描述")
    else:
        print("  未找到题目描述文件, 使用纯AST特征")
    return problem_texts


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("AST-KC Discovery Pipeline")
    print("=" * 60)
    
    # ---- 数据集选择 ----
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'Code-S'
    print(f'数据集: {dataset}')
    
    if dataset == 'BePKT':
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        DATA_DIR = os.path.join(BASE_DIR, 'data/BePKT')
        OUTPUT_PATH = os.path.join(DATA_DIR, 'ast_kc_mapping.json')
        CODE_FILE = 'CodeStates.csv'
        MAIN_FILE = 'MainTable.csv'
        N_CLUSTERS = None  # 动态确定
        MAX_SAMPLES = 20000
    else:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        DATA_DIR = os.path.join(BASE_DIR, 'data/Code-S')
        OUTPUT_PATH = os.path.join(DATA_DIR, 'ast_kc_mapping.json')
        CODE_FILE = 'CodeStates_5000.csv'
        MAIN_FILE = 'MainTable_5000.csv'
        N_CLUSTERS = None  # 动态确定
        MAX_SAMPLES = None
    MAX_DEPTH = 4  # 加深1层以获取更多结构信息
    D_MODEL = 128  # 增大维度以编码更丰富的特征
    BATCH_SIZE = 32
    EPOCHS = 20
    # ---- KC映射超参数 ----
    KC_THRESHOLD_RATIO = 0.03    # 软分配阈值 (权重摊薄了, 需降低)
    KC_MIN_VOTES = 0.5           # 软分配最小累计权重
    KC_MIN_PER_PROBLEM = 2       # 每道题最少保留KC数
    KC_MAX_PER_PROBLEM = 5       # 每道题最多保留KC数
    KC_SOFT_TEMPERATURE = 0.5    # 软分配温度: 越小越硬, 越大越软
    TEXT_FEATURE_ALPHA = 0.7     # AST向量权重, 文本向量=1-ALPHA
    # ========================
    
    # ---- 1. 加载代码数据 ----
    print("\n[1/8] 加载代码数据...")
    code_df = pd.read_csv(f'{DATA_DIR}/{CODE_FILE}')
    code_df = code_df.dropna(subset=['Code'])
    if MAX_SAMPLES and len(code_df) > MAX_SAMPLES:
        code_df = code_df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"  (采样 {MAX_SAMPLES} 条)")
    print(f"  {len(code_df)} 段代码")
    
    # 加载MainTable获取ProblemID关联
    main_df = pd.read_csv(f'{DATA_DIR}/{MAIN_FILE}')
    
    # ---- 动态确定聚类数 ----
    num_problems = main_df['ProblemID'].nunique()
    N_CLUSTERS = max(10, min(int(num_problems ** 0.5) * 3, 80))
    print(f"  {num_problems} 道题, 动态设定 N_CLUSTERS={N_CLUSTERS}")
    # ============================
    
    # ---- 2. 解析AST ----
    print("\n[2/8] 解析AST...")
    # BePKT是C/C++代码, Code-S是Java代码
    if dataset == 'BePKT':
        parser = init_c_parser()
    else:
        parser = init_java_parser()
    
    code_subtrees = {}  # CodeStateID → [子树列表]
    all_subtrees = []   # 所有子树的聚合列表
    
    for _, row in tqdm(code_df.iterrows(), total=len(code_df), desc="解析"):
        cid = row['CodeStateID']
        code = row['Code']
        
        root = parse_code_to_ast(parser, code)
        if root is None:
            continue
        
        subtrees = extract_subtrees(root, max_depth=MAX_DEPTH)
        code_subtrees[cid] = subtrees
        all_subtrees.extend(subtrees)
    
    print(f"  共提取 {len(all_subtrees)} 个子树")
    print(f"  平均每段代码 {len(all_subtrees)/max(len(code_subtrees),1):.1f} 个子树")
    
    # ---- 3. 构建词表 ----
    print("\n[3/8] 构建AST类型词表...")
    type_vocab = build_type_vocab(all_subtrees)
    print(f"  词表大小: {len(type_vocab)} 种AST节点类型")
    
    # ---- 4. 组织训练数据 ----
    print("\n[4/8] 组织对比学习数据...")
    
    # 加载Score (Code-S的Score在V30子目录中)
    score_map = {}
    score_path = os.path.join(DATA_DIR, 'CodeScore.csv')
    if not os.path.exists(score_path):
        score_path = os.path.join(DATA_DIR, 'V30', 'CodeScore_5000.csv')
    if os.path.exists(score_path):
        score_df = pd.read_csv(score_path)
        for _, r in score_df.iterrows():
            try:
                score_map[r['CodeStateID']] = float(r['Score']) / 50.0
            except:
                pass
    
    # 关联CodeStateID到ProblemID和Score
    merged = main_df.merge(code_df[['CodeStateID']], on='CodeStateID', how='inner')
    
    problem_groups = defaultdict(list)
    for _, row in merged.iterrows():
        pid = row['ProblemID']
        cid = row['CodeStateID']
        # 优先用CodeScore里的分数
        if cid in score_map:
            score_val = score_map[cid]
        else:
            score_val = 0.5
        if cid in code_subtrees:
            problem_groups[pid].append((
                score_val, 
                code_subtrees[cid], 
                cid
            ))
    
    dataset = CodePairDataset(problem_groups, type_vocab)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_subtrees(b, max_subtrees=30)
    )
    print(f"  {len(dataset)} 个对比样本对")
    
    # ---- 5. 训练Transformer ----
    print("\n[5/8] 训练重要性打分器...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")
    
    scorer = ImportanceScorer(
        vocab_size=len(type_vocab),
        d_model=D_MODEL,
    ).to(device)
    
    train_importance_scorer(scorer, dataloader, epochs=EPOCHS)
    
    # ---- 6. 加权聚类 ----
    print("\n[6/8] 加权聚类形成KCs...")
    
    # 加载题目文本特征 (辅助聚类)
    problem_text_vecs = load_problem_texts(DATA_DIR)
    ALPHA = TEXT_FEATURE_ALPHA

    # 提取所有子树的向量 (AST + 文本)
    subtree_vectors = []
    importance_weights = []
    
    scorer.eval()
    with torch.no_grad():
        for pid, group in problem_groups.items():
            text_vec = torch.tensor(problem_text_vecs.get(str(pid), np.zeros(D_MODEL)),
                                    device=scorer.device, dtype=torch.float)
            for score, subtrees, cid in group:
                for sub in subtrees:
                    ids = subtree_to_ids(sub, type_vocab)
                    ids_tensor = torch.tensor([ids], device=scorer.device)
                    ast_vec = scorer.embedding(ids_tensor).mean(dim=1).squeeze()
                    # AST + 文本 混合特征
                    combined = ALPHA * ast_vec + (1 - ALPHA) * text_vec
                    subtree_vectors.append(combined.cpu())
                    importance_weights.append(score)
    
    kc_centers = np.zeros((N_CLUSTERS, D_MODEL))
    kc_patterns = {}
    valid_subtree_indices = []
    
    if subtree_vectors:
        vecs = torch.stack(subtree_vectors)  # [N, D_MODEL]
        weights = torch.tensor(importance_weights)
        
        # 过滤NaN和Inf
        valid = ~torch.isnan(vecs.sum(dim=1)) & ~torch.isinf(vecs.sum(dim=1))
        valid_indices = torch.where(valid)[0].tolist()
        
        if len(valid_indices) < N_CLUSTERS * 2:
            print(f"  警告: 有效向量不足 ({len(valid_indices)}), 跳过聚类")
        else:
            vecs = vecs[valid]
            w = weights[valid] + 1e-6
            valid_subtree_indices = valid_indices
            
            kc_labels, kc_centers = cluster_into_kcs(
                vecs, w, n_clusters=N_CLUSTERS
            )
            # 将标签映射回原始子树
            n_subtrees = len(all_subtrees)
            full_kc_labels = -1 * np.ones(n_subtrees, dtype=int)
            for idx, label in zip(valid_indices, kc_labels):
                if idx < n_subtrees:
                    full_kc_labels[idx] = label
            
            kc_patterns = analyze_kc_patterns(full_kc_labels, all_subtrees, N_CLUSTERS)
            
            print(f"  形成 {len(kc_patterns)} 个KCs:")
            for kc_id, pattern in sorted(kc_patterns.items()):
                print(f"    KC_{kc_id:2d}: {pattern['description']:20s} ({pattern['num_members']}个子树)")
    
    # ---- 7. 生成KC映射 ----
    print("\n[7/8] 生成Problem→KC映射...")
    
    problem_kc_map = {}
    
    # 有效KC的ID列表 (KMeans可能有空簇)
    kc_ids_sorted = sorted(kc_patterns.keys())

    if len(kc_ids_sorted) == 0:
        print("  警告: 无有效KC, 使用默认映射")
        for pid in problem_groups:
            problem_kc_map[str(pid)] = list(range(min(N_CLUSTERS, 10)))
    else:
        for pid, group in problem_groups.items():
            # 所有代码都参与投票 (包含低分代码有助于识别易错KC)
            text_vec = torch.tensor(problem_text_vecs.get(str(pid), np.zeros(D_MODEL)), dtype=torch.float)
            kc_weighted = Counter()
            for score, subtrees, cid in group:
                w = max(0.6, score)  # 分数越高权重越大, 最低0.6
                for sub in subtrees:
                    ids = subtree_to_ids(sub, type_vocab)
                    with torch.no_grad():
                        ids_tensor = torch.tensor([ids], device=scorer.device)
                        ast_vec = scorer.embedding(ids_tensor).mean(dim=1).squeeze().cpu()
                    combined_vec = (ALPHA * ast_vec + (1 - ALPHA) * text_vec).numpy().flatten()
                    dists = np.linalg.norm(kc_centers - combined_vec, axis=1)
                    # 软分配: 每棵子树按距离softmax权重贡献给所有KC
                    soft_w = np.exp(-dists / KC_SOFT_TEMPERATURE)
                    soft_w = soft_w / (soft_w.sum() + 1e-10)
                    for kc_id in kc_ids_sorted:
                        kc_weighted[kc_id] += w * soft_w[kc_id]
            
            # 加权投票过滤: 保留权重占比≥KC_THRESHOLD_RATIO的KC
            if kc_weighted:
                total_weight = sum(kc_weighted.values())
                threshold = max(KC_MIN_VOTES, total_weight * KC_THRESHOLD_RATIO)
                filtered_kcs = [k for k, v in kc_weighted.items() if v >= threshold]
                # 至少保留KC_MIN_PER_PROBLEM个, 最多保留KC_MAX_PER_PROBLEM个
                if len(filtered_kcs) < KC_MIN_PER_PROBLEM:
                    filtered_kcs = [k for k, _ in kc_weighted.most_common(KC_MIN_PER_PROBLEM)]
                if len(filtered_kcs) > KC_MAX_PER_PROBLEM:
                    filtered_kcs = [k for k, _ in kc_weighted.most_common(KC_MAX_PER_PROBLEM)]
                problem_kc_map[str(pid)] = sorted(filtered_kcs)
            else:
                problem_kc_map[str(pid)] = [0]
    
    # ---- 计算KC相似度矩阵 (用于图卷积) ----
    print("\n[8/8] 计算KC相似度矩阵...")
    n_kcs = len(kc_ids_sorted)
    kc_sim = np.eye(n_kcs, dtype=np.float32)

    # 基于type_distribution计算KC间余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    type_vectors = []
    type_vocab_all = set()
    for kc_id in kc_ids_sorted:
        for t in kc_patterns[kc_id]['type_distribution']:
            type_vocab_all.add(t)
    type_list = sorted(type_vocab_all)
    type_to_idx = {t: i for i, t in enumerate(type_list)}
    for kc_id in kc_ids_sorted:
        vec = np.zeros(len(type_list), dtype=np.float32)
        for t, cnt in kc_patterns[kc_id]['type_distribution'].items():
            vec[type_to_idx[t]] = cnt
        type_vectors.append(vec)
    type_vectors = np.array(type_vectors)
    sim_matrix = cosine_similarity(type_vectors)  # [N, N]

    # 建立kc_id到索引的映射
    kc_to_idx = {kid: i for i, kid in enumerate(kc_ids_sorted)}

    # 结合共现相似度: 两KC出现在同一道题的频率
    co_occur = np.zeros((n_kcs, n_kcs), dtype=np.float32)
    for pid, kcs in problem_kc_map.items():
        idxs = [kc_to_idx[k] for k in kcs]
        for i in idxs:
            for j in idxs:
                co_occur[i][j] += 1.0
    row_sum = co_occur.sum(axis=1, keepdims=True) + 1e-8
    co_occur_norm = co_occur / row_sum

    # 最终相似度 = 0.5*AST类型 + 0.5*共现
    final_sim = 0.5 * sim_matrix + 0.5 * co_occur_norm

    # k-NN稀疏化 + 实值相似度: 每个KC只保留最相似的 K 个邻居, 权重为相似度
    K = max(5, n_kcs // 10)
    adj = np.zeros((n_kcs, n_kcs), dtype=np.float32)
    for i in range(n_kcs):
        sim_row = final_sim[i]
        top_k = np.argpartition(-sim_row, K)[:K]
        adj[i, top_k] = sim_row[top_k]  # 保留实值相似度, 而非二值化
    adj = np.maximum(adj, adj.T)
    adj = np.maximum(adj, np.eye(n_kcs))
    print(f"  KC图: {n_kcs}节点, {int((adj > 0).sum() - n_kcs)}条边 (自环除外), K={K}")

    # ---- 保存 (kc_list按kc_id排序, 与adj的行对应) ----
    mapping = {
        "num_kcs": n_kcs,
        "kc_ids": kc_ids_sorted,
        "kc_list": [kc_patterns[kid]["description"] for kid in kc_ids_sorted],
        "problem_kc_map": problem_kc_map,
        "kc_code_patterns": {str(k): v for k, v in kc_patterns.items()},
        "kc_adjacency": adj.tolist(),
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完成! 结果已保存到 {OUTPUT_PATH}")
    print(f"   {len(kc_patterns)} 个KCs")
    print(f"   {len(problem_kc_map)} 道题已映射")


if __name__ == '__main__':
    main()
