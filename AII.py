import pandas as pd
import numpy as np
import RRDA
# import seaborn as sns
# import matplotlib.pyplot as plt
import util
import draw_utils
from params import V
import warnings
import argparse
from params import RRDA_AI_indicater_1

warnings.filterwarnings('ignore')


# def get_choice(df: pd.DataFrame,
#                item: str,
#                item_id='indicator_2',
#                choice_id='indicator_3',
#                score_id='score'):
#     # Get the choices and scores for a given item
#     df_choices = df[df[item_id] == item]
#     choices = df_choices[choice_id].tolist()
#     scores = df_choices[score_id].tolist()

#     ch_scs = {}
#     for choice, score in zip(choices, scores):
#         ch_scs[choice] = score

#     ch_scs[np.nan] = 0.  # 空值默认为0
#     print(f'item: {item}, choices: {ch_scs}')
#     return ch_scs


# def get_scores(df_idx: pd.DataFrame, df_std: pd.DataFrame,
#                item: str) -> np.ndarray:
#     # Get the scores for a specific item in df_std
#     np_item = df_std[item].to_numpy()
#     choices = get_choice(df_idx, item)
#     get_value = lambda x: choices.get(x)
#     get_value_vec = np.vectorize(get_value, otypes=[np.float64])
#     np_scores = get_value_vec(np_item)

#     return np_scores

def get_choice(df: pd.DataFrame,
               item: str,
               item_id='indicator_2',
               choice_id='indicator_3',
               score_id='score'):
    # Get the choices and scores for a given item
    df_choices = df[df[item_id] == item]
    choices = df_choices[choice_id].tolist()
    scores = df_choices[score_id].tolist()

    ch_scs = {}
    for choice, score in zip(choices, scores):
        ch_scs[choice] = score

    # Ensure missing values are handled
    ch_scs[np.nan] = 0.  # 空值默认为0
    ch_scs[None] = 0.  # None值默认为0
    return ch_scs


def get_scores(df_idx: pd.DataFrame, df_std: pd.DataFrame,
               item: str) -> np.ndarray:
    # Get the scores for a specific item in df_std
    np_item = df_std[item].to_numpy()
    choices = get_choice(df_idx, item)
    get_value = lambda x: choices.get(x, 0)  # 如果找不到值，默认返回0
    get_value_vec = np.vectorize(get_value, otypes=[np.float64])
    np_scores = get_value_vec(np_item)

    return np_scores


def get_additional_score(df_std: pd.DataFrame) -> np.ndarray:
    # 20260207 patch: 针对受抚养情况的case进行特殊处理
    additional_scores = np.array([0.] * df_std.shape[0])
    condition_1 = (df_std['2.受抚养情况'] == '由父母双方抚养') | \
                  (df_std['2.受抚养情况'] == '单亲家庭子女，父母离异，由父母双方抚养孩子')
    condition_2 = (df_std['2.受抚养情况'] == '单亲家庭子女，父母一方去世，由父母其中一方独自抚养孩子') | \
                  (df_std['2.受抚养情况'] == '单亲家庭子女，父母离异，由父母其中一方独自抚养孩子')
    condition_3 = df_std['2.受抚养情况'] == '父母双方均不具有或均未承担抚养责任'

    additional_scores[condition_1] += 0.0
    additional_scores[condition_2] += 20.0
    additional_scores[condition_3] += 40.0
    
    return additional_scores


def AII(df_std,
        save=True,
        df_idx=None,
        RRDA_score=None,
        indices: np.ndarray = None,
        work_dir=None,
        poverty_perc=None,
        patch=False, # 20260115补丁：针对受抚养情况的case
        ):

    OPTIM = False

    if patch:
        df_std = util.data_clean(df_std, df_idx)

    if indices is not None:
        df_std = df_std.loc[indices]

    # Get unique items
    items = list(set(df_idx['indicator_2'].tolist()))

    # Calculate scores for each item
    final_res = np.array([0] * df_std.shape[0])
    scores = np.array([0.] * df_std.shape[0])
    for item in items:
        try:
            score = get_scores(df_idx, df_std, item)
            # 特殊处理某些指标
            if item == '家庭经济困难类型':
                final_res[score == 999] = 3
            elif item == '举报提供虚假信息证明材料':
                final_res[score == -1] = -1
            elif item == '2.受抚养情况':
                # 20260207 patch: 针对受抚养情况的case进行特殊处理
                scores += score + get_additional_score(df_std)
            else:
                scores += score
        except KeyError:
            pass

    # Calculate total scores

    if RRDA_score:
        rrda_score = 0.0 if OPTIM else RRDA_score
        scores += df_std['家庭经济困难度指数'].to_numpy() * rrda_score
    
    df_std['困难度分数'] = scores

    df_std = util.pov_assign(df_std, V, poverty_perc=poverty_perc)

    if work_dir is not None:
        try:
            # util.visualization(df_std,
            #                    V,
            #                    filepath=work_dir + '/AII_困难度分值真实分布.png')
            draw_utils.kde_plot(
                df_std['实际认定结果'].tolist(),
                df_std['困难度分数'].tolist(),
                V,
                filepath=work_dir + '/AII_困难度分值分布核密度图.png',
            )
            draw_utils.draw_bar(
                df_std['实际认定结果'].tolist(),
                df_std['困难度分数'].tolist(),
                V,
                filepath=work_dir + '/AII_困难度分值分布柱状图.png',
            )
            # draw_utils.draw_matrix(
            #     df_std,
            #     '实际认定结果',
            #     '算法认定结果',
            #     filepath=work_dir + '/AII_混淆矩阵.png',
            # )
        except Exception as e:
            print(e)

    df_std = df_std.drop(columns=['original_index'])
    if save:
        df_std.to_csv(work_dir + '/AII.csv', encoding='utf_8_sig', index=False)

    return df_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AII')
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
                        default='D:/Working/code/AII_PyQT/input/新指标体系.csv',
                        help='指标体系文件')
    args = parser.parse_args()
    # work_dir = 'data/base'
    # all_item_std_file = work_dir + '/raw_500_modified.csv'

    # df_std = RRDA.run_RRDA(args.dataset_file)

    df_std = pd.read_csv(args.dataset_file, encoding='utf_8_sig')
    

    poverty_perc = {
        '特别困难': 0.4,
        '困难': 0.3,
        '一般困难': 0.2,
        '不困难': 0.1,
    }

    df_idx = pd.read_csv(args.zbtx_file)
    A_I, A_II = util.get_AI_AII(
        df_idx,
        # RRDA_AI_indicater_1=RRDA_AI_indicater_1,
    )
    # print(f'A_I: {A_I}')
    # print(f'A_II: {A_II}')

    # RRDA_score = A_II[RRDA_AI_indicater_1]['家庭经济困难度指数'] * A_I[
    #     RRDA_AI_indicater_1] * 100
    # print(f'RRDA_score: {RRDA_score}')

    df_std = AII(
        df_std,
        save=True,
        df_idx=df_idx,
        work_dir=args.work_dir,
        # RRDA_score=RRDA_score,
        poverty_perc=poverty_perc,
    )
    acc = util.calc_acc(df_std, '算法认定结果', '实际认定结果')
    if acc:
        print(f'一致性：{acc * 100 :.2f}%')
    print('AII 计算完成')

    new_file = args.dataset_file.replace('.csv', '_AII.csv')
    df_std.to_csv(new_file, encoding='utf_8_sig', index=False)
