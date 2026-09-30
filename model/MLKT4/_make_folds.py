"""
_make_folds.py — 生成真 5 折 (学生级) 数据划分

用途: 把现有 MSDGKT/q_a_{train,valid,test}_detailed.csv 合并为全量,
      按 SubjectID 随机分成 5 组 (固定种子可复现), 每折:
        test  = 组 i        (20%)
        valid = 组 (i+1)%5  (20%)
        train = 其余 3 组    (60%)
      写入 data/{data_dir}/MSDGKT/fold{i}/q_a_{train,valid,test}_detailed.csv

之后用 train_kc_mastery.py:
  python train_kc_mastery.py --data_dir BePKT --emb_size 256 --batch_size 12 --max_iter 100 --fold -1
  (--fold=-1 遍历 5 折, 每折读 fold{i}/ 数据 + 独立种子 seed+fold)
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))  # model/MLKT4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='BePKT', choices=['BePKT', 'Code-S'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    base = os.path.join(BASE, '..', '..', 'data', args.data_dir, 'MSDGKT')

    # 合并现有 train+valid+test = 全部保留学生
    parts = []
    for split in ['train', 'valid', 'test']:
        p = os.path.join(base, 'q_a_%s_detailed.csv' % split)
        if not os.path.exists(p):
            raise FileNotFoundError('缺少: %s' % p)
        df = pd.read_csv(p)
        df['_src'] = split
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    users = np.array(sorted(full['SubjectID'].unique()))
    print('合并数据: %d 行, %d 学生' % (len(full), len(users)))

    # 固定种子打乱 → 分成 5 组 (学生级, 每个学生只属于一组)
    rng = np.random.RandomState(args.seed)
    users_sh = rng.permutation(users)
    groups = np.array_split(users_sh, 5)
    print('各组学生数: %s' % [len(g) for g in groups])

    for i in range(5):
        test_u = set(groups[i].tolist())
        valid_u = set(groups[(i + 1) % 5].tolist())
        train_u = set(users.tolist()) - test_u - valid_u
        train_df = full[full['SubjectID'].isin(train_u)].drop(columns=['_src'])
        valid_df = full[full['SubjectID'].isin(valid_u)].drop(columns=['_src'])
        test_df = full[full['SubjectID'].isin(test_u)].drop(columns=['_src'])
        out = os.path.join(base, 'fold%d' % i)
        os.makedirs(out, exist_ok=True)
        for name, part in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
            part.to_csv(os.path.join(out, 'q_a_%s_detailed.csv' % name),
                        index=False, quoting=csv.QUOTE_NONNUMERIC)
        print('fold%d: train=%d行/%d人  valid=%d行/%d人  test=%d行/%d人' % (
            i, len(train_df), train_df['SubjectID'].nunique(),
            len(valid_df), valid_df['SubjectID'].nunique(),
            len(test_df), test_df['SubjectID'].nunique()))

    print('\n完成: 5 折数据已写入 %s/fold{0..4}/' % base)
    print('随后运行: python train_kc_mastery.py --data_dir %s --emb_size 256 --batch_size 12 --max_iter 100 --fold -1' % args.data_dir)


if __name__ == '__main__':
    main()
