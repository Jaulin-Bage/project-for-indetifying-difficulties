import pandas as pd
import numpy as np
import RRDA
import util
import draw_utils
from params import V, RRDA_AI_indicater_1
import warnings
import argparse

warnings.filterwarnings('ignore')


def optim_R_all(df: pd.DataFrame,
                df_idx: pd.DataFrame,
                V: list,
                noliner_transform=False) -> dict:

    def calculate_R3(df_subset: pd.DataFrame, V: list,
                     noliner_transform: bool) -> list:
        R3 = [
            float(df_subset[df_subset['实际认定结果'] == res].shape[0]) /
            float(df_subset.shape[0]) for res in V
        ]
        if noliner_transform:  # 执行非线性变换，使R3分布更紧凑
            R3_np = np.array(R3)**3
            R3_np /= R3_np.sum()
            R3 = R3_np.tolist()
        return R3

    R_all = {}
    for id2, group in df_idx.groupby('indicator_2'):
        R2 = {}
        for id3 in group['indicator_3'].unique():
            df_id3 = df[df[id2] == id3]
            if not df_id3.empty:
                R2[id3] = calculate_R3(df_id3, V, noliner_transform)

        # 处理缺失值
        df_id3_nan = df[df[id2].isna()]
        if not df_id3_nan.empty:
            R2[np.nan] = calculate_R3(df_id3_nan, V, noliner_transform)

        R_all[id2] = R2

    return R_all


def create_R_simple(row, R_all, df_idx):
    R_simple = {}
    for id2 in df_idx['indicator_2'].unique():
        id3 = row.get(id2)
        if id3 in R_all.get(id2, {}):
            R_simple[id2] = R_all[id2][id3]
    return R_simple


def calc_B_prime(R_simple: dict, A_II: dict, V: list, indicator_1: list,
                 df_idx: pd.DataFrame):
    B_prime = {}
    for id1 in indicator_1:
        indicator_2 = df_idx.loc[df_idx['indicator_1'] == id1,
                                 'indicator_2'].unique()
        B = np.zeros(len(V))
        for id2 in indicator_2:
            try:
                # print(id1, id2, A_II[id1][id2], R_simple[id2])
                B += A_II[id1][id2] * np.array(R_simple[id2])
            except KeyError:
                pass
        B_prime[id1] = B
    return B_prime


def calc_B(B_prime: dict, A_I: dict, V: list, indicator_1: list):
    B = np.zeros(len(V))
    for id1 in indicator_1:
        try:
            B += A_I[id1] * B_prime[id1]
        except KeyError:
            pass
    return B


def calc_pov_score(B: list, S: list):
    score = sum(b * s for b, s in zip(B, S))
    assert 0 <= score <= 100, f'score={score}\nB={B}\nS={S}'
    return score


def calc_all_std_pov(df_std: pd.DataFrame, R_all, V, S, df_idx):
    A_I, A_II = util.get_AI_AII(df_idx)

    indicator_1 = df_idx['indicator_1'].drop_duplicates().tolist()
    Bs = np.zeros((len(df_std), len(V)), dtype=np.float32)

    scores = RRDA.data_clean(df_std).tolist()

    for idx, (_, row) in enumerate(df_std.iterrows()):
        if scores[idx] != -1:  # 未认定
            if scores[idx] == 0:
                Bs[idx, -1] = 1.
            elif scores[idx] == 100:
                Bs[idx, 0] = 1.
            else:
                exit()
            continue
        R_simple = create_R_simple(row, R_all, df_idx)
        # print(R_simple)
        B_prime = calc_B_prime(R_simple, A_II, V, indicator_1, df_idx)
        # print(B_prime)
        B = calc_B(B_prime, A_I, V, indicator_1)
        # print(B)
        Bs[idx] = B
        scores[idx] = calc_pov_score(B, S)

    return Bs, scores


def trans_t_to_R(t):
    assert t >= 0 and t <= 1

    def k(t):
        return t / 3 - 1 / 6

    def b(t):
        return 1 / 4 - 3 * k(t) / 2

    def f(x, t):
        return x * k(t) + b(t)

    return np.array([f(0, t), f(1, t), f(2, t), f(3, t)][::-1])


# def init_R_all(df_idx: pd.DataFrame):
#     df_norm_scs = pd.read_csv('data/指标体系_v1_expected.csv')
#     R1 = {}
#     for id2, df_id2 in df_idx.groupby('indicator_2'):
#         R2 = {}
#         for id3, t in df_norm_scs[df_norm_scs['indicator_2'] == id2].groupby(
#                 'indicator_3')['normalized_score'].first().items():
#             # print(id3, t)
#             R2[id3] = trans_t_to_R(t)
#         # R2[np.nan] = np.array([0.25, 0.25, 0.25, 0.25])
#         R2[np.nan] = np.array([0., 0., 0., 1.])
#         R1[id2] = R2

#     rrdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
#     rrdas_lev_str = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#     for rrda_lev, rrda in zip(rrdas_lev_str, rrdas):
#         R1['家庭经济困难度指数等级'][rrda_lev] = trans_t_to_R(rrda)

#     return R1


def FCE_AII(
    df_std=None,
    zbtx_file=None,
    save=True,
    indices: np.ndarray = None,
    work_dir=None,
    poverty_perc=None,
    with_RRDA=True,
):

    df_idx = pd.read_csv(zbtx_file)

    if with_RRDA:
        # 将 家庭经济情况,家庭经济困难度指数等级,9 加入df_idx
        for i in range(0, 10):
            new_row = {
                'indicator_1': RRDA_AI_indicater_1,
                'indicator_2': '家庭经济困难度指数',
                'indicator_3': str(i),
            }
            new_row_df = pd.DataFrame([new_row])
            df_idx = pd.concat([df_idx, new_row_df], ignore_index=True)

    S = [100, 66, 33, 0]  # 评价得分矩阵

    # df_std['家庭经济困难度指数'] = (df_std['家庭经济困难度指数'] * 10).astype(int).astype(str)
    
    R_all = optim_R_all(df_std, df_idx, V, noliner_transform=False)
    # pprint.pprint(R_all)

    if indices is not None:
        df_std = df_std.loc[indices]

    Bs, scores = calc_all_std_pov(df_std, R_all, V, S, df_idx)
    df_std['困难度分数'] = scores

    df_std = util.pov_assign(df_std, V, poverty_perc)

    if work_dir:
        # util.visualization(df_std,
        #                    V,
        #                    filepath=work_dir + '/FCEAII_困难度分值真实分布.png')
        draw_utils.kde_plot(
            df_std['实际认定结果'].tolist(),
            df_std['困难度分数'].tolist(),
            V,
            filepath=work_dir + '/FCEAII_困难度分值分布核密度图.png',
        )
        draw_utils.draw_bar(
            df_std['实际认定结果'].tolist(),
            df_std['困难度分数'].tolist(),
            V,
            filepath=work_dir + '/FCEAII_困难度分值分布柱状图.png',
        )
    df_std = df_std.drop(columns=['original_index'])
    if save:
        df_std.to_csv(work_dir + '/FCE_AII.csv',
                      encoding='utf_8_sig',
                      index=False)

    return df_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCE_AII')
    parser.add_argument('--work_dir',
                        type=str,
                        default='tmp',
                        help='工作目录')
    parser.add_argument('--dataset_file',
                        type=str,
                        default='tmp/新模拟数据.csv',
                        help='数据集文件')
    parser.add_argument('--zbtx_file',
                        type=str,
                        default='tmp/整合版量化指标体系.csv',
                        help='指标体系文件')
    args = parser.parse_args()
    # work_dir = 'data/base'
    # dataset_file = work_dir + '/raw_500_modified.csv'
    # zbtx_file = 'data/指标体系_v1.csv'

    # df_std = RRDA.run_RRDA(args.dataset_file)
    df_std = pd.read_csv(args.dataset_file, encoding='utf_8_sig')

    poverty_perc = {
        '特别困难': 0.4,
        '困难': 0.3,
        '一般困难': 0.2,
        '不困难': 0.1,
    }

    df_std = FCE_AII(
        df_std,
        args.zbtx_file,
        save=True,
        work_dir=args.work_dir,
        with_RRDA=False,
    )
    acc = util.calc_acc(df_std, '算法认定结果', '实际认定结果')
    print(f'一致性：{acc * 100 :.2f}%')