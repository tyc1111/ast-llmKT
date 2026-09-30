"""
data_save_Code_S&F.py 的改动
===============================
把 Skills 替换为 KC_mask
"""

# ===== 改动1: 在文件顶部新增加载KC映射 =====
import json

# 加载AST-KC映射 (新增)
KC_MAPPING_PATH = '../../data/Code-S/ast_kc_mapping.json'
if os.path.exists(KC_MAPPING_PATH):
    with open(KC_MAPPING_PATH, 'r', encoding='utf-8') as f:
        kc_mapping = json.load(f)
    NUM_KCS = kc_mapping['num_kcs']
    problem_kc_map = kc_mapping['problem_kc_map']
    print(f"加载KC映射: {NUM_KCS} 个KCs, {len(problem_kc_map)} 道题")
else:
    print("警告: 未找到KC映射, 回退到原有Skills")
    kc_mapping = None


# ===== 改动2: 选择列时用KC_mask替换Skills =====
# 原来:
# columns = ['SubjectID', 'Skills', 'ProblemID', 'Requirement', 'Score', ...]
# 
# 改为:
if kc_mapping is not None:
    # 用KC_mask
    all_data['KC_mask'] = all_data['ProblemID'].apply(
        lambda pid: problem_kc_map.get(str(int(pid)), [])
    )
    columns = ['SubjectID', 'KC_mask', 'ProblemID', 'Requirement', 'Score', ...]
else:
    # 回退到Skills
    columns = ['SubjectID', 'Skills', 'ProblemID', 'Requirement', 'Score', ...]


# ===== 改动3: 在训练数据循环中 =====
# 原来:
# eval(temp1.iloc[i]['Skills'])
#
# 改为:
if kc_mapping is not None:
    kc_mask = temp1.iloc[i]['KC_mask']  # 已经是list
    # KC_mask 格式: [3, 5, 8, ...] 表示涉及KC_3, KC_5, KC_8
else:
    kc_mask = eval(temp1.iloc[i]['Skills'])
