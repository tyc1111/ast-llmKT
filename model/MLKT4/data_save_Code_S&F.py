import sys
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
SEQ_LEN = 200

# ============================================================
# 将5D Reason扩展为7D (新增逻辑错误数和语法错误数)
# ============================================================



# Load data
main_table = pd.read_csv('../../data/Code-S/MainTable_5000.csv')

# ==== (新增) 加载AST-KC映射 ====
KC_MAPPING_PATH = '../../data/Code-S/ast_kc_mapping.json'
if os.path.exists(KC_MAPPING_PATH):
    with open(KC_MAPPING_PATH, 'r', encoding='utf-8') as f:
        kc_mapping = json.load(f)
    NUM_KCS = kc_mapping['num_kcs']
    problem_kc_map = kc_mapping['problem_kc_map']
    print(f'加载KC映射: {NUM_KCS} 个KCs, {len(problem_kc_map)} 道题')
else:
    print('警告: 未找到KC映射, 回退到原有Skills')
    problem_kc_map = None
# ================================

code_table = pd.read_csv('../../data/Code-S/CodeStates_5000.csv')
score_table = pd.read_csv('../../data/Code-S/V30/CodeScore_5000.csv')
reason_table = pd.read_csv('../../data/Code-S/V30/CodeReason_5000.csv')
# Merge tables
code_table = code_table.dropna(subset=['Code'])
main_table = main_table.merge(code_table, how='left', on='CodeStateID')



# sys.exit()
# Clean data by dropping rows where 'Score' is NaN and reset index
main_table.dropna(subset=['Score'], inplace=True)

# Step 1: Record CodeStateIDs where Score == 1.0
score_one_code_state_ids = main_table[main_table['Score'] == 1.0]['CodeStateID']
#
main_table = main_table.drop('Score', axis=1)
main_table = main_table.merge(score_table, how='right', on='CodeStateID')
main_table = main_table.merge(reason_table, how='left', on='CodeStateID')

# # Delete rows where 'Reason' is NaN
main_table = main_table.dropna(subset=['Reason'], inplace=False)

column_name = 'Score'
main_table[column_name] = pd.to_numeric(main_table[column_name], errors='coerce')
main_table[column_name] = main_table[column_name] / 50

# Step 2: Manually set new Score to 1.0 for rows where original Score was 1.0
main_table.loc[main_table['CodeStateID'].isin(score_one_code_state_ids), column_name] = 1.0

main_table.reset_index(drop=True, inplace=True)

# Define functions


def genr_new_qid(QuestionId):
    return problem2id[QuestionId]

def genr_label(Score):
    return 1.0 if Score > 0.5 else 0

# Load mappings
with open('../../data/Code-S/user2id', 'r', encoding='utf-8') as fi:
    user2id = eval(fi.read())
with open('../../data/Code-S/problem2id', 'r', encoding='utf-8') as fi:
    problem2id = eval(fi.read())

# Apply transformations
main_table = main_table.dropna(subset=['CodeStateID'])
main_table['ProblemID'] = main_table['ProblemID'].apply(genr_new_qid)
# main_table['Score'] = main_table['Score'].apply(genr_label)

# ==== 用KC_mask替换Skills ====
if problem_kc_map is not None:
    # 将ProblemID映射回原始ID再查KC映射
    rev_problem2id = {v: k for k, v in problem2id.items()}
    main_table['KC_mask'] = main_table['ProblemID'].apply(
        lambda pid: problem_kc_map.get(str(int(rev_problem2id.get(int(pid), pid))), [])
    )
    columns = ['SubjectID', 'ProblemID', 'Score', 'ServerTimestamp', 'Code', 'Reason', 'KC_mask']
else:
    columns = ['SubjectID', 'Skills', 'ProblemID', 'Score', 'ServerTimestamp', 'Code', 'Reason']
all_data = main_table[columns]

# Prepare for modeling
all_user = np.array(all_data['SubjectID'])
user = sorted(list(set(all_user)))


# Split data
train_all_id, temp_id = train_test_split(user, test_size=0.2, random_state=5)
train_id = np.array(train_all_id)
test_id, valid_id = train_test_split(temp_id, test_size=0.5, random_state=5)

# Initialize training data
q_a_train = []

for item in tqdm(train_id):
    # print(item)
    idx = all_data[all_data['SubjectID'] == item].index.tolist() # 找到对应学生的index
    temp1 = all_data.iloc[idx].sort_values(by='ServerTimestamp') # 找到对应学生的行
    if len(temp1) < 2:
        continue
    
    # ==== 使用原始5D Reason ====
    reason_5d_list = [eval(r) for r in temp1['Reason'].values[:SEQ_LEN]]
    
    prev_timestamp = None
    for i in range(len(temp1)):
        if i >= SEQ_LEN:
            break
        
        # ==== 使用KC_mask替换Skills ====
        if 'KC_mask' in all_data.columns:
            kc_data = temp1.iloc[i]['KC_mask']
            if isinstance(kc_data, list):
                kc_value = kc_data
            else:
                kc_value = eval(kc_data) if isinstance(kc_data, str) else []
        else:
            kc_value = eval(temp1.iloc[i]['Skills'])
        # ================================
        
        # 计算时间间隔
        from datetime import datetime
        ts = temp1.iloc[i]['ServerTimestamp']
        try:
            cur_time = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S').timestamp()
        except:
            cur_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()
        time_interval = 0.0 if prev_timestamp is None else (cur_time - prev_timestamp) / 60
        prev_timestamp = cur_time
        
        q_a_train.append([
            user2id[temp1.iloc[i]['SubjectID']],
            int(temp1.iloc[i]['ProblemID']),
            kc_value,
            reason_5d_list[i],
            float(temp1.iloc[i]['Score']),
            temp1.iloc[i]['ServerTimestamp'],
            round(time_interval, 4),
        ])

# Save to CSV
kc_col_name = 'KC_mask' if 'KC_mask' in all_data.columns else 'Skills'
df = pd.DataFrame(q_a_train, columns=[
    'SubjectID','ProblemID', kc_col_name, 'Reason', 'Label',
    'ServerTimestamp', 'TimeInterval'
])
os.makedirs('../../data/Code-S/MSDGKT', exist_ok=True)
df.to_csv('../../data/Code-S/MSDGKT/q_a_train_detailed.csv', index=False)

q_a_test = []

# Process data for each user
for item in tqdm(test_id):  # 使用测试数据集
    idx = all_data[all_data['SubjectID'] == item].index.tolist()
    temp1 = all_data.iloc[idx].sort_values(by='ServerTimestamp')
    if len(temp1) < 2:
        continue
    
    # ==== 使用原始5D Reason ====
    reason_5d_list = [eval(r) for r in temp1['Reason'].values[:SEQ_LEN]]
    
    prev_timestamp = None
    for i in range(len(temp1)):
        if i >= SEQ_LEN:
            break
        
        if 'KC_mask' in all_data.columns:
            kc_data = temp1.iloc[i]['KC_mask']
            if isinstance(kc_data, list):
                kc_value = kc_data
            else:
                kc_value = eval(kc_data) if isinstance(kc_data, str) else []
        else:
            kc_value = eval(temp1.iloc[i]['Skills'])
        
        from datetime import datetime
        ts = temp1.iloc[i]['ServerTimestamp']
        try:
            cur_time = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S').timestamp()
        except:
            cur_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()
        time_interval = 0.0 if prev_timestamp is None else (cur_time - prev_timestamp) / 60
        prev_timestamp = cur_time
        
        q_a_test.append([
            user2id[temp1.iloc[i]['SubjectID']],
            int(temp1.iloc[i]['ProblemID']),
            kc_value,
            reason_5d_list[i],
            float(temp1.iloc[i]['Score']),
            temp1.iloc[i]['ServerTimestamp'],
            round(time_interval, 4),
        ])

kc_col_name = 'KC_mask' if 'KC_mask' in all_data.columns else 'Skills'
df_test = pd.DataFrame(q_a_test, columns=[
    'SubjectID','ProblemID', kc_col_name, 'Reason', 'Label',
    'ServerTimestamp', 'TimeInterval'
])
df_test.to_csv('../../data/Code-S/MSDGKT/q_a_test_detailed.csv', index=False)



q_a_valid = []

# Process data for each user
for item in tqdm(valid_id):  # 使用验证数据集
    idx = all_data[all_data['SubjectID'] == item].index.tolist()
    temp1 = all_data.iloc[idx].sort_values(by='ServerTimestamp')
    if len(temp1) < 2:
        continue
    
    # ==== 使用原始5D Reason ====
    reason_5d_list = [eval(r) for r in temp1['Reason'].values[:SEQ_LEN]]
    
    prev_timestamp = None
    for i in range(len(temp1)):
        if i >= SEQ_LEN:
            break
        
        if 'KC_mask' in all_data.columns:
            kc_data = temp1.iloc[i]['KC_mask']
            if isinstance(kc_data, list):
                kc_value = kc_data
            else:
                kc_value = eval(kc_data) if isinstance(kc_data, str) else []
        else:
            kc_value = eval(temp1.iloc[i]['Skills'])
        
        from datetime import datetime
        ts = temp1.iloc[i]['ServerTimestamp']
        try:
            cur_time = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S').timestamp()
        except:
            cur_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()
        time_interval = 0.0 if prev_timestamp is None else (cur_time - prev_timestamp) / 60
        prev_timestamp = cur_time
        
        q_a_valid.append([
            user2id[temp1.iloc[i]['SubjectID']],
            int(temp1.iloc[i]['ProblemID']),
            kc_value,
            reason_5d_list[i],
            float(temp1.iloc[i]['Score']),
            temp1.iloc[i]['ServerTimestamp'],
            round(time_interval, 4),
        ])

kc_col_name = 'KC_mask' if 'KC_mask' in all_data.columns else 'Skills'
df_valid = pd.DataFrame(q_a_valid, columns=[
    'SubjectID','ProblemID', kc_col_name, 'Reason', 'Label',
    'ServerTimestamp', 'TimeInterval'
])
df_valid.to_csv('../../data/Code-S/MSDGKT/q_a_valid_detailed.csv', index=False)
print('complete')


