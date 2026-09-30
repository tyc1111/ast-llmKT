import logging as log
import os
import numpy as np
import torch
from torch.utils import data
import pandas as pd

import copy
import ast


_CODE_CACHE = {}


def _load_code_assets(data_root):
    """加载 code_map.csv + code_feat.npz → ((SubjectID,idx)->feat行号, features[矩阵]).
    缺失文件时返回 (None, None), 全流程退化为全零特征(off 路径不变)。"""
    key = os.path.abspath(data_root)
    if key in _CODE_CACHE:
        return _CODE_CACHE[key]
    cm_p = os.path.join(data_root, 'code_map.csv')
    cf_p = os.path.join(data_root, 'code_feat.npz')
    if not (os.path.exists(cm_p) and os.path.exists(cf_p)):
        print('  [code_feat] 未找到 code_map.csv/code_feat.npz → 代码特征全零')
        _CODE_CACHE[key] = (None, None)
        return _CODE_CACHE[key]
    cm = pd.read_csv(cm_p)
    d = np.load(cf_p, allow_pickle=True)
    feats = d['features'].astype('float32')
    cidx = {csid: i for i, csid in enumerate(d['code_state_id'])}
    lookup = {}
    for r in cm.itertuples(index=False):
        j = cidx.get(r.CodeStateID)
        if j is not None:
            lookup[(int(r.SubjectID), int(r.idx))] = j
    print(f'  [code_feat] 载入 {len(lookup)} 条映射, F={feats.shape[1]}')
    _CODE_CACHE[key] = (lookup, feats)
    return _CODE_CACHE[key]


_TOK_CACHE = {}


def _load_code_tokens(data_root):
    """加载 code_tokens.npz → ((SubjectID,idx)->行号, tokens[N,L]); 缺失返回 (None,None)"""
    key = os.path.abspath(data_root)
    if key in _TOK_CACHE:
        return _TOK_CACHE[key]
    p = os.path.join(data_root, 'code_tokens.npz')
    if not os.path.exists(p):
        print('  [code_tok] 未找到 code_tokens.npz → token特征全0')
        _TOK_CACHE[key] = (None, None)
        return _TOK_CACHE[key]
    d = np.load(p, allow_pickle=True)
    toks = d['tokens']
    cidx = {csid: i for i, csid in enumerate(d['code_state_id'])}
    cm = pd.read_csv(os.path.join(data_root, 'code_map.csv'))
    lookup = {}
    for r in cm.itertuples(index=False):
        j = cidx.get(r.CodeStateID)
        if j is not None:
            lookup[(int(r.SubjectID), int(r.idx))] = j
    print(f'  [code_tok] 载入 {len(lookup)} 条映射, L={toks.shape[1]}, V={len(d["vocab"])}')
    _TOK_CACHE[key] = (lookup, toks)
    return _TOK_CACHE[key]


class KTDataset(data.Dataset):
    def __init__(self, args,problem_number, batch_size, seq_len, mode):

        self.problem_number = problem_number

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.path = args.data_dir
        self.model = args.model
        self.device = args.device
        self.kc_mode = getattr(args, 'kc_mode', 'kcmask')  # kcmask=AST-KC, skills=人工Skills
        self.num_kcs = int(getattr(args, 'num_kcs', 0) or 0)
        # [code] 代码张量开关: 默认关。关闭时不加载 code_map.csv/code_feat.npz/code_tokens.npz,
        # 也不构建 *_code_tensor(35维AST特征) / *_code_tok(128 token), 只返回空的占位张量。
        # 需要时(Stage A/B 实验 或 _diag_code_use.py)加 --code_assets 打开。
        self.use_code_assets = bool(getattr(args, 'code_assets', False))
        # [7] 折目录支持: --fold>=0 时从 fold{fold}/ 读取对应数据划分 (真5折)
        self.fold = getattr(args, 'fold', -1)
        self._base = '../../data/' + self.path + '/' + getattr(args, 'data_subdir', self.model)
        if self.fold >= 0:
            self._base += '/fold%d' % self.fold




        self.mode = mode
        if not self.use_code_assets:
            print('  [code] 未开启 --code_assets → 跳过代码张量构建(空占位, 不读 code_map/npz)')
        log.info('Processing data...')
        self.process(self.path, self.mode)
        if self.mode == 'Train':
            self.data_size = len(self.data_dict['train_problem_tensor'])
        elif self.mode == 'Valid':
            self.data_size = len(self.data_dict['valid_problem_tensor'])
        elif self.mode == 'Test':
            self.data_size = len(self.data_dict['test_problem_tensor'])

        log.info('Processing data done!')

    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        if self.mode == 'Train':
            return self.data_dict['train_problem_tensor'][i], self.data_dict['train_skills_tensor'][i],\
                self.data_dict['train_label_tensor'][i],self.data_dict['train_note_tensor'][i],\
                self.data_dict['train_reasons_tensor'][i],self.data_dict['train_time_stamp_tensor'][i],\
                self.data_dict['train_debug_tensor'][i], self.data_dict['train_code_tensor'][i],\
                self.data_dict['train_code_tok'][i], *self._dual_masks_for_item('train', i)
        elif self.mode == 'Valid':
            return self.data_dict['valid_problem_tensor'][i], self.data_dict['valid_skills_tensor'][i],\
                self.data_dict['valid_label_tensor'][i],self.data_dict['valid_note_tensor'][i],\
                self.data_dict['valid_reasons_tensor'][i],self.data_dict['valid_time_stamp_tensor'][i],\
                self.data_dict['valid_debug_tensor'][i], self.data_dict['valid_code_tensor'][i],\
                self.data_dict['valid_code_tok'][i], *self._dual_masks_for_item('valid', i)
        elif self.mode == 'Test':
            return self.data_dict['test_problem_tensor'][i], self.data_dict['test_skills_tensor'][i],\
                self.data_dict['test_label_tensor'][i],self.data_dict['test_note_tensor'][i],\
                self.data_dict['test_reasons_tensor'][i],self.data_dict['test_time_stamp_tensor'][i],\
                self.data_dict['test_debug_tensor'][i], self.data_dict['test_code_tensor'][i],\
                self.data_dict['test_code_tok'][i], *self._dual_masks_for_item('test', i)

    def _dual_masks_for_item(self, split, index):
        if self.kc_mode != 'dualkc':
            return ()
        return (self.data_dict[f'{split}_query_kc_mask'][index],
                self.data_dict[f'{split}_event_kc_mask'][index])

    def gen_dual_kc_mask(self, subject_ids, kc_lists):
        """Build dense [student, sequence, KC] masks; KC 0 is a valid ID."""
        if self.num_kcs <= 0:
            raise ValueError('dualkc mode requires args.num_kcs > 0')
        grouped = {}
        for sid, values in zip(subject_ids, kc_lists):
            grouped.setdefault(int(sid), []).append(values)
        rows = []
        for sid, seq in grouped.items():
            arr = np.zeros((self.seq_len, self.num_kcs), dtype=np.float32)
            for step, raw in enumerate(seq[:self.seq_len]):
                values = ast.literal_eval(raw) if isinstance(raw, str) else raw
                for kid in values:
                    kid = int(kid)
                    if not 0 <= kid < self.num_kcs:
                        raise ValueError(f'KC ID {kid} outside [0,{self.num_kcs}) for student {sid}')
                    arr[step, kid] = 1.0
            rows.append(arr)
        if not rows:
            return torch.zeros(0, self.seq_len, self.num_kcs, dtype=torch.float32)
        return torch.from_numpy(np.stack(rows, axis=0))

    def read_dual_mask_columns(self, table, subject_ids):
        if self.kc_mode != 'dualkc':
            return None, None
        required = {'KCQuery_mask', 'KC_mask'}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f'dualkc input is missing columns: {sorted(missing)}')
        query = self.gen_dual_kc_mask(subject_ids, list(table['KCQuery_mask']))
        event = self.gen_dual_kc_mask(subject_ids, list(table['KC_mask']))
        return query, event

    def pad_and_convert(self, lists, target_length, padding_value=0):
        """
        Pad each sublist in the given list of lists to the target length and convert to a Torch IntTensor.

        Args:
        lists (list of list of int): The list of lists to pad.
        target_length (int): The desired length of each sublist.
        padding_value (int, optional): The value to use for padding. Defaults to 0.

        Returns:
        torch.IntTensor: A tensor of shape (len(lists), target_length) where each sublist is padded.
        """
        padded_lists = []
        for sublist in lists:
            # Truncate the list if it's longer than the target length or pad it if it's shorter
            padded_list = sublist[:target_length] + [padding_value] * (target_length - len(sublist))
            padded_lists.append(padded_list)

        # Convert the list of lists into a tensor of type int
        # print(padded_lists[0][0])
        return torch.tensor(padded_lists, dtype=torch.int32)
    # def pad_and_convert(self, lists, target_length, padding_value=0):
    #     """
    #     Pad each sublist in the given list of lists to the target length and convert to a Torch IntTensor.
    #
    #     Args:
    #     lists (list of list of int): The list of lists to pad.
    #     target_length (int): The desired length of each sublist.
    #     padding_value (int, optional): The value to use for padding. Defaults to 0.
    #
    #     Returns:
    #     torch.IntTensor: A tensor of shape (len(lists), target_length) where each sublist is padded.
    #     """
    #     print("\n" + "=" * 80)
    #     print("🔍 pad_and_convert 调试信息")
    #     print("=" * 80)
    #     print(f"总列表数量: {len(lists)}")
    #     print(f"目标长度: {target_length}")
    #     print(f"填充值: {padding_value}")
    #
    #     padded_lists = []
    #
    #     for idx, sublist in enumerate(lists):
    #         print(f"\n--- 处理第 {idx} 个列表 ---")
    #         print(f"原始 sublist 类型: {type(sublist)}")
    #         print(f"原始 sublist 长度: {len(sublist) if isinstance(sublist, list) else 'N/A'}")
    #         print(f"原始 sublist 内容: {sublist}")
    #
    #         # 检查 sublist 中的每个元素
    #         if isinstance(sublist, list):
    #             for i, element in enumerate(sublist):
    #                 if not isinstance(element, (int, float)):
    #                     print(f"⚠️  警告: 第 {i} 个元素不是数字!")
    #                     print(f"   元素值: {element}")
    #                     print(f"   元素类型: {type(element)}")
    #                     print(f"   这会导致 torch.tensor 报错!")
    #
    #         # Truncate the list if it's longer than the target length or pad it if it's shorter
    #         try:
    #             padded_list = sublist[:target_length] + [padding_value] * (target_length - len(sublist))
    #             print(f"填充后长度: {len(padded_list)}")
    #             print(f"填充后内容: {padded_list}")
    #
    #             # 再次检查填充后的列表
    #             for i, element in enumerate(padded_list):
    #                 if not isinstance(element, (int, float)):
    #                     print(f"❌ 错误: 填充后第 {i} 个元素仍然不是数字!")
    #                     print(f"   元素值: {element}")
    #                     print(f"   元素类型: {type(element)}")
    #
    #             padded_lists.append(padded_list)
    #         except Exception as e:
    #             print(f"❌ 填充过程出错: {e}")
    #             raise
    #
    #     print("\n" + "=" * 80)
    #     print(f"即将转换为 tensor 的数据:")
    #     print(f"总共 {len(padded_lists)} 个列表")
    #
    #     # 检查第一个列表的详细信息
    #     if padded_lists:
    #         print(f"\n第一个列表详情:")
    #         print(f"  长度: {len(padded_lists[0])}")
    #         print(f"  内容: {padded_lists[0]}")
    #         print(f"  第一个元素: {padded_lists[0][0]}, 类型: {type(padded_lists[0][0])}")
    #
    #     # Convert the list of lists into a tensor of type int
    #     try:
    #         print("\n尝试转换为 torch.tensor...")
    #         result = torch.tensor(padded_lists, dtype=torch.int32)
    #         print(f"✅ 转换成功! Tensor shape: {result.shape}")
    #         return result
    #     except TypeError as e:
    #         print(f"\n❌❌❌ torch.tensor 转换失败! ❌❌❌")
    #         print(f"错误信息: {e}")
    #         print(f"\n详细分析:")
    #
    #         # 找出有问题的元素
    #         for i, lst in enumerate(padded_lists):
    #             for j, element in enumerate(lst):
    #                 if isinstance(element, str):
    #                     print(f"  问题位置: 第 {i} 个列表的第 {j} 个元素")
    #                     print(f"  问题元素: '{element}' (类型: {type(element).__name__})")
    #                 elif not isinstance(element, (int, float)):
    #                     print(f"  问题位置: 第 {i} 个列表的第 {j} 个元素")
    #                     print(f"  问题元素: {element} (类型: {type(element).__name__})")
    #
    #         raise

    def gen_list(self, subject_ids, requirements, max_length, column_type=None):

        if column_type == 'Label':
            subject_labels = {}
            for subject_id, requirement in zip(subject_ids, requirements):
                if subject_id not in subject_labels:
                    subject_labels[subject_id] = []

                subject_labels[subject_id].append(requirement)

            result = []
            note = []
            for subject_id in subject_labels.keys():
                requirements = copy.deepcopy(subject_labels[subject_id])  # 使用深拷贝
                temp_note = copy.deepcopy(subject_labels[subject_id])  # 使用深拷贝

                if len(requirements) < max_length:
                    padding_length = max_length - len(requirements)

                    requirements.extend([0] * padding_length)
                    temp_note.extend([-1] * padding_length)
                    # print([-1] * (max_length - len(requirements)))

                result.append(requirements)

                note.append(temp_note)

            return result, note

        elif column_type in ['Problem','ErrorNum']:
            subject_problems = {}

            for subject_id, problems in zip(subject_ids, requirements):
                if subject_id not in subject_problems:
                    subject_problems[subject_id] = []

                subject_problems[subject_id].append(problems)

            result = []
            note = []
            for subject_id in subject_problems.keys():

                requirements = subject_problems[subject_id]

                if len(requirements) < max_length:
                    requirements.extend([0] * (max_length - len(requirements)))
                result.append(requirements)
            return result

        elif column_type == "TimeInterval":
            subject_TimeInterval = {}

            for subject_id, ServerTimestamp in zip(subject_ids, requirements):
                if subject_id not in subject_TimeInterval:
                    subject_TimeInterval[subject_id] = []

                subject_TimeInterval[subject_id].append(ServerTimestamp)

            result = []
            for subject_id in subject_TimeInterval.keys():

                requirements = subject_TimeInterval[subject_id]

                if len(requirements) < max_length:
                    requirements.extend([0] * (max_length - len(requirements)))


                result.append(requirements)

            return result

        elif column_type == 'Code':
            subject_requirements = {}
            # 遍历 SubjectID 和 Requirement 的组合，保持原始顺序
            for subject_id, requirement in zip(subject_ids, requirements):
                # 如果 SubjectID 已经在字典中，添加 Requirement，否则创建新列表
                if subject_id not in subject_requirements:
                    subject_requirements[subject_id] = []

                subject_requirements[subject_id].append(requirement)

            # 维护原始顺序，使用一个列表来存储每个学生的 Requirement 列表
            result = []
            for subject_id in subject_requirements.keys():
                # 获取该 SubjectID 对应的 Requirement 列表
                requirements = subject_requirements[subject_id]

                # 如果 Requirement 数量少于 max_length，则用 "空语句" 填充

                if len(requirements) < max_length:
                    requirements.extend(['public int nullFunction() {}'] * (max_length - len(requirements)))
                    # print(requirements)

                # 将填充后的列表添加到结果中
                result.append(requirements)
            return result
        elif column_type == 'Skills':
            subject_requirements = {}
            for subject_id, requirement in zip(subject_ids, requirements):
                if subject_id not in subject_requirements:
                    subject_requirements[subject_id] = []
                subject_requirements[subject_id].append(requirement)

            # 初始化一个空的三维张量
            result = torch.empty(0, 0, max_length, dtype=torch.int32)

            # 处理每个 SubjectID 的需求并按期望的形状拼接
            for subject_id in subject_requirements.keys():
                requirements = subject_requirements[subject_id]
                requirements_tensor = self.pad_and_convert(requirements, max_length)

                if requirements_tensor.shape[0] < self.seq_len:
                    padding_size = self.seq_len - requirements_tensor.shape[0]
                    # 在第二维度（列）填充0
                    pad_tensor = torch.zeros([padding_size, max_length])

                    requirements_tensor = torch.cat([requirements_tensor, pad_tensor], dim=0)

                # 为了满足形状[num_SubjectID, Seq_len, 6]，需要增加一维
                requirements_tensor = requirements_tensor.unsqueeze(0)  # 这里将形状从[Seq_len, 6]变为[1, Seq_len, 6]
                if result.shape[1:] == (0, max_length):  # 如果result还是初始空状态，设置正确的形状
                    result = requirements_tensor
                else:
                    result = torch.cat([result, requirements_tensor], dim=0)  # 沿着第一个维度（SubjectID维度）拼接
            return result

    @staticmethod
    def _count_subj(subject_ids):
        """学生数(去重, 保持首次出现顺序) —— 关闭 code_assets 时用于生成占位张量的首维"""
        return len(dict.fromkeys(int(s) for s in subject_ids))

    def gen_code_feat(self, subject_ids, max_length):
        """按 SubjectID 首次出现顺序 + 组内行序, 组装[num_subj, seq_len, F]代码特征张量;
        行序必须与 CSV 行序一致(即其它张量的分组顺序)。缺失资产时全零。
        关闭 code_assets 时返回 [num_subj, 0, 0] 占位(形状合法、可直接索引)。"""
        if not self.use_code_assets:
            return torch.zeros(self._count_subj(subject_ids), 0, 0, dtype=torch.float32)
        data_root = '../../data/' + self.path
        lookup, feats = _load_code_assets(data_root)
        F = feats.shape[1] if feats is not None else 35
        subj_feats, cnt = {}, {}
        for sid in subject_ids:
            sid = int(sid)
            k = cnt.get(sid, 0)
            cnt[sid] = k + 1
            if sid not in subj_feats:
                subj_feats[sid] = []
            j = lookup.get((sid, k)) if lookup is not None else None
            subj_feats[sid].append(feats[j] if j is not None else np.zeros(F, dtype='float32'))
        rows = []
        for sid in subj_feats:
            arr = np.stack(subj_feats[sid], 0).astype('float32')
            if arr.shape[0] < self.seq_len:
                arr = np.concatenate([arr, np.zeros((self.seq_len - arr.shape[0], F), dtype='float32')], 0)
            else:
                arr = arr[:self.seq_len]
            rows.append(arr)
        if not rows:
            return torch.zeros(0, self.seq_len, F, dtype=torch.float32)
        return torch.tensor(np.stack(rows, 0), dtype=torch.float32)

    def gen_code_tok(self, subject_ids):
        """按 SubjectID 首次出现顺序+组内行序, 组装[num_subj, seq_len, L] token 序列(long);
        缺失资产时全零。关闭 code_assets 时返回 [num_subj, 0, 0] 占位。"""
        if not self.use_code_assets:
            return torch.zeros(self._count_subj(subject_ids), 0, 0, dtype=torch.long)
        data_root = '../../data/' + self.path
        lookup, toks = _load_code_tokens(data_root)
        L = toks.shape[1] if toks is not None else 128
        subj, cnt = {}, {}
        for sid in subject_ids:
            sid = int(sid)
            k = cnt.get(sid, 0)
            cnt[sid] = k + 1
            if sid not in subj:
                subj[sid] = []
            j = lookup.get((sid, k)) if lookup is not None else None
            subj[sid].append(toks[j] if j is not None else np.zeros(L, dtype='int16'))
        rows = []
        for sid in subj:
            arr = np.stack(subj[sid], 0).astype('int64')
            if arr.shape[0] < self.seq_len:
                arr = np.concatenate([arr, np.zeros((self.seq_len - arr.shape[0], L), dtype='int64')], 0)
            else:
                arr = arr[:self.seq_len]
            rows.append(arr)
        if not rows:
            return torch.zeros(0, self.seq_len, L, dtype=torch.long)
        return torch.tensor(np.stack(rows, 0), dtype=torch.long)

    def process(self, path, mode='Train'):

        if mode == 'Train':
            print("Training data reading...")

            train_table = pd.read_csv(self._base + '/q_a_train_detailed.csv')
            train_problemid = list(train_table['ProblemID'])
            train_subjectid = list(train_table['SubjectID'])
            train_query_kc_mask, train_event_kc_mask = self.read_dual_mask_columns(train_table, train_subjectid)
            train_answer = list(train_table['Label'])
            train_time_stamp = list(train_table['TimeInterval'])
            train_reasons = list(train_table['Reason'].apply(eval))
            
            # ==== 兼容: 读取KC_mask或Skills (按 kc_mode 切换) ====
            if 'KC_mask' in train_table.columns and self.kc_mode == 'kcmask':
                train_skills = list(train_table['KC_mask'].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                ))
                print(f'  使用KC_mask (kc_mode={self.kc_mode})')
            else:
                train_skills = list(train_table['Skills'].apply(eval))
                print('  使用人工Skills')
            # ================================

            # ==== (新增) 加载全局纠错能力(兼容旧版CSV) ====
            if 'GlobalDebugAbility' in train_table.columns:
                train_debug_dict = {}
                for sid, dval in zip(train_subjectid, train_table['GlobalDebugAbility']):
                    if sid not in train_debug_dict:
                        train_debug_dict[sid] = float(dval)
            else:
                train_debug_dict = None

            train_label_list, train_note_list = self.gen_list(train_subjectid, train_answer, max_length=self.seq_len,
                                                              column_type="Label")  # [num_stu,seq_len]，前者用0填充，后者用1填充
            train_problem_list = self.gen_list(train_subjectid, train_problemid, max_length=self.seq_len,
                                               column_type='Problem')  # [num_stu,seq_len]
            train_time_stamp_list = self.gen_list(train_subjectid,train_time_stamp,max_length=self.seq_len,column_type='TimeInterval')


            train_label_tensor = torch.FloatTensor(train_label_list)  # torch.Size([num_stu,seq_len])
            train_note_tensor = torch.FloatTensor(train_note_list)
            train_problem_tensor = torch.IntTensor(train_problem_list)  # torch.Size([num_stu,seq_len])


            if self.kc_mode == 'dualkc':
                train_skills_tensor = torch.zeros(len(train_problem_tensor), self.seq_len, 1, dtype=torch.int32)
            else:
                train_skills_tensor = self.gen_list(train_subjectid, train_skills, max_length=8,
                                                    column_type='Skills')  # torch.Size([num_stu,seq_len,k])
            # ==== 修改: 7D Reason (原5D → 7D) ====
            train_reasons_tensor = self.gen_list(train_subjectid,train_reasons,max_length=5,
                                                 column_type='Skills')
            train_time_stamp_tensor = torch.FloatTensor(train_time_stamp_list)
            train_code_tensor = self.gen_code_feat(train_subjectid, max_length=self.seq_len)
            train_code_tok = self.gen_code_tok(train_subjectid)
            # ==== (新增) 全局纠错能力张量 ====
            if train_debug_dict is not None:
                unique_subjects = sorted(set(train_subjectid))
                train_debug_tensor = torch.FloatTensor([train_debug_dict[s] for s in unique_subjects]).unsqueeze(1)
            else:
                train_debug_tensor = torch.zeros(len(set(train_subjectid)), 1)

            self.data_dict = {
                'train_problem_tensor': train_problem_tensor,  # [num_stu,seq_len]
                'train_skills_tensor': train_skills_tensor,
                'train_label_tensor': train_label_tensor,  # [num_stu,seq_len]
                'train_note_tensor': train_note_tensor,
                'train_reasons_tensor': train_reasons_tensor,
                'train_time_stamp_tensor':train_time_stamp_tensor,
                'train_debug_tensor': train_debug_tensor,  # [num_stu, 1]
                'train_code_tensor': train_code_tensor,
                'train_code_tok': train_code_tok,
                'train_query_kc_mask': train_query_kc_mask,
                'train_event_kc_mask': train_event_kc_mask
            }
            return self.data_dict

        elif mode == 'Valid':
            print("Validation data reading...")

            valid_table = pd.read_csv(self._base + '/q_a_valid_detailed.csv')
            valid_problemid = list(valid_table['ProblemID'])
            valid_subjectid = list(valid_table['SubjectID'])
            valid_query_kc_mask, valid_event_kc_mask = self.read_dual_mask_columns(valid_table, valid_subjectid)
            valid_answer = list(valid_table['Label'])
            valid_time_stamp = list(valid_table['TimeInterval'])
            valid_reasons = list(valid_table['Reason'].apply(eval))
            if 'KC_mask' in valid_table.columns and self.kc_mode == 'kcmask':
                valid_skills = list(valid_table['KC_mask'].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                ))
            else:
                valid_skills = list(valid_table['Skills'].apply(eval))

            # ==== (新增) 加载全局纠错能力(兼容旧版CSV) ====
            if 'GlobalDebugAbility' in valid_table.columns:
                valid_debug_dict = {}
                for sid, dval in zip(valid_subjectid, valid_table['GlobalDebugAbility']):
                    if sid not in valid_debug_dict:
                        valid_debug_dict[sid] = float(dval)
            else:
                valid_debug_dict = None

            valid_label_list, valid_note_list = self.gen_list(valid_subjectid, valid_answer, max_length=self.seq_len,
                                                              column_type="Label")  # [num_stu,seq_len]，前者用0填充，后者用1填充
            valid_problem_list = self.gen_list(valid_subjectid, valid_problemid, max_length=self.seq_len,
                                               column_type='Problem')  # [num_stu,seq_len]
            valid_time_stamp_list = self.gen_list(valid_subjectid, valid_time_stamp, max_length=self.seq_len,
                                                  column_type='TimeInterval')

            valid_label_tensor = torch.FloatTensor(valid_label_list)  # torch.Size([num_stu,seq_len])
            valid_note_tensor = torch.FloatTensor(valid_note_list)
            valid_problem_tensor = torch.IntTensor(valid_problem_list)  # torch.Size([num_stu,seq_len])

            if self.kc_mode == 'dualkc':
                valid_skills_tensor = torch.zeros(len(valid_problem_tensor), self.seq_len, 1, dtype=torch.int32)
            else:
                valid_skills_tensor = self.gen_list(valid_subjectid, valid_skills, max_length=8,
                                                    column_type='Skills')  # torch.Size([num_stu,seq_len,k])
            # ==== 修改: 7D Reason ====
            valid_reasons_tensor = self.gen_list(valid_subjectid, valid_reasons, max_length=5,
                                                 column_type='Skills')
            valid_time_stamp_tensor = torch.FloatTensor(valid_time_stamp_list)
            valid_code_tensor = self.gen_code_feat(valid_subjectid, max_length=self.seq_len)
            valid_code_tok = self.gen_code_tok(valid_subjectid)
            # ==== (新增) 全局纠错能力张量 ====
            if valid_debug_dict is not None:
                unique_subjects = sorted(set(valid_subjectid))
                valid_debug_tensor = torch.FloatTensor([valid_debug_dict[s] for s in unique_subjects]).unsqueeze(1)
            else:
                valid_debug_tensor = torch.zeros(len(set(valid_subjectid)), 1)

            self.data_dict = {
                'valid_problem_tensor': valid_problem_tensor,  # [num_stu,seq_len]
                'valid_skills_tensor': valid_skills_tensor,
                'valid_label_tensor': valid_label_tensor,  # [num_stu,seq_len]
                'valid_note_tensor': valid_note_tensor,
                'valid_reasons_tensor': valid_reasons_tensor,
                'valid_time_stamp_tensor': valid_time_stamp_tensor,
                'valid_debug_tensor': valid_debug_tensor,  # [num_stu, 1]
                'valid_code_tensor': valid_code_tensor,
                'valid_code_tok': valid_code_tok,
                'valid_query_kc_mask': valid_query_kc_mask,
                'valid_event_kc_mask': valid_event_kc_mask
            }
            return self.data_dict
        elif mode == "Test":
            print("Test data reading...")

            test_table = pd.read_csv(self._base + '/q_a_test_detailed.csv')
            test_problemid = list(test_table['ProblemID'])
            test_subjectid = list(test_table['SubjectID'])
            test_query_kc_mask, test_event_kc_mask = self.read_dual_mask_columns(test_table, test_subjectid)
            test_answer = list(test_table['Label'])
            test_time_stamp = list(test_table['TimeInterval'])
            test_reasons = list(test_table['Reason'].apply(eval))
            if 'KC_mask' in test_table.columns and self.kc_mode == 'kcmask':
                test_skills = list(test_table['KC_mask'].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                ))
            else:
                test_skills = list(test_table['Skills'].apply(eval))

            # ==== (新增) 加载全局纠错能力(兼容旧版CSV) ====
            if 'GlobalDebugAbility' in test_table.columns:
                test_debug_dict = {}
                for sid, dval in zip(test_subjectid, test_table['GlobalDebugAbility']):
                    if sid not in test_debug_dict:
                        test_debug_dict[sid] = float(dval)
            else:
                test_debug_dict = None

            test_label_list, test_note_list = self.gen_list(test_subjectid, test_answer, max_length=self.seq_len,
                                                            column_type="Label")  # [num_stu,seq_len]，前者用0填充，后者用1填充
            test_problem_list = self.gen_list(test_subjectid, test_problemid, max_length=self.seq_len,
                                              column_type='Problem')  # [num_stu,seq_len]
            test_time_stamp_list = self.gen_list(test_subjectid, test_time_stamp, max_length=self.seq_len,
                                                 column_type='TimeInterval')

            test_label_tensor = torch.FloatTensor(test_label_list)  # torch.Size([num_stu,seq_len])
            test_note_tensor = torch.FloatTensor(test_note_list)
            test_problem_tensor = torch.IntTensor(test_problem_list)  # torch.Size([num_stu,seq_len])

            if self.kc_mode == 'dualkc':
                test_skills_tensor = torch.zeros(len(test_problem_tensor), self.seq_len, 1, dtype=torch.int32)
            else:
                test_skills_tensor = self.gen_list(test_subjectid, test_skills, max_length=8,
                                                   column_type='Skills')  # torch.Size([num_stu,seq_len,k])
            # ==== 修改: 7D Reason ====
            test_reasons_tensor = self.gen_list(test_subjectid, test_reasons, max_length=5,
                                                column_type='Skills')
            test_time_stamp_tensor = torch.FloatTensor(test_time_stamp_list)
            test_code_tensor = self.gen_code_feat(test_subjectid, max_length=self.seq_len)
            test_code_tok = self.gen_code_tok(test_subjectid)
            # ==== (新增) 全局纠错能力张量 ====
            if test_debug_dict is not None:
                unique_subjects = sorted(set(test_subjectid))
                test_debug_tensor = torch.FloatTensor([test_debug_dict[s] for s in unique_subjects]).unsqueeze(1)
            else:
                test_debug_tensor = torch.zeros(len(set(test_subjectid)), 1)

            self.data_dict = {
                'test_problem_tensor': test_problem_tensor,  # [num_stu,seq_len]
                'test_skills_tensor': test_skills_tensor,
                'test_label_tensor': test_label_tensor,  # [num_stu,seq_len]
                'test_note_tensor': test_note_tensor,
                'test_reasons_tensor': test_reasons_tensor,
                'test_time_stamp_tensor': test_time_stamp_tensor,
                'test_debug_tensor': test_debug_tensor,  # [num_stu, 1]
                'test_code_tensor': test_code_tensor,
                'test_code_tok': test_code_tok,
                'test_query_kc_mask': test_query_kc_mask,
                'test_event_kc_mask': test_event_kc_mask
            }
            return self.data_dict









