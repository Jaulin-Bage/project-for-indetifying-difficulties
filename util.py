import pandas as pd

def calc_acc(df, col1, col2):
    try:
        acc = df[df[col1] == df[col2]].shape[0] / df.shape[0]
        return acc
    except ZeroDivisionError:
        print('数据集为空')
    except KeyError:
        print('KeyError')
    return None


def pov_assign(df_std, V: list, poverty_perc: dict = {}):
    '''
    困难度认定
    '''
    if poverty_perc:
        poverty_nums = [
            int(df_std.shape[0] * poverty_perc.get(v, 0)) for v in V
        ]
    elif '实际认定结果' in df_std.columns:
        poverty_nums = [df_std['实际认定结果'].tolist().count(v) for v in V]
    else:
        # 默认分布，防止报错
        print(
            "Warning: '实际认定结果' column missing and no poverty_perc provided. Using default distribution."
        )
        default_perc = {'特别困难': 0.1, '困难': 0.2, '一般困难': 0.3, '不困难': 0.4}
        poverty_nums = [
            int(df_std.shape[0] * default_perc.get(v, 0)) for v in V
        ]

    index = list(range(df_std.shape[0]))
    df_std['original_index'] = index
    excepted_res = []
    for pov_level, level_num in zip(V, poverty_nums):
        excepted_res += [pov_level] * level_num

    # 如果分配的数量不足，总数不足时进行填充
    remaining_count = df_std.shape[0] - len(excepted_res)
    if remaining_count > 0:
        # 假设填充到最后一个级别（可以根据需求修改）
        excepted_res += [V[-1]] * remaining_count

    df_std_sorted = df_std.sort_values(by='困难度分数', ascending=False)
    df_std_sorted['算法认定结果'] = excepted_res
    df_std_reset = df_std_sorted.sort_values(by='original_index')
    df_zeros_cond = (df_std_reset['困难度分数'] == 0)
    df_std_reset.loc[df_zeros_cond,
                     '算法认定结果'] = ([V[3]] *
                                  df_std_reset[df_zeros_cond].shape[0])
    return df_std_reset


def get_AI_AII(
    df_idx,
    RRDA_AI_indicater_1,
):
    """ 
        根据指标体系v1获取A_I, A_II 
        df: 指标体系v1的DataFrame
        RRDA_AI_indicater_1: RRDA对应的一级指标
    """
    # 按照indicator2分组
    # df = pd.read_csv(zbtx_path)
    df_grouped = df_idx.groupby('indicator_2')
    # 获取每个组的第一个元素
    df_first = df_grouped.first()
    new_df = df_first.reset_index()
    new_df.drop(
        columns=['Unnamed: 0', 'indicator_3', 'score', 'normalized_score'],
        inplace=True,
        errors='ignore')

    new_df_grouped = new_df.groupby('indicator_1')
    # 获取new_df_grouped每组score_max的和
    A_I = new_df_grouped['score_max'].sum().to_dict()
    # 每一项的值都除以100
    A_I = {k: v / 100 for k, v in A_I.items()}

    # 当RRDA_AI_indicater_1不在A_I中时，添加RRDA_AI_indicater_1
    if RRDA_AI_indicater_1 in A_I:
        A_I[RRDA_AI_indicater_1] = 0
    A_I[RRDA_AI_indicater_1] = 1 - sum(A_I.values())

    RRDA_score = 1 - sum(A_I.values())
    A_I[RRDA_AI_indicater_1] += RRDA_score

    # 根据new_df获取A_II
    A_II = {}
    for k in A_I.keys():
        df_tmp = new_df[new_df['indicator_1'] == k]
        A_II[k] = {
            indicator_2:
            float(df_tmp[df_tmp['indicator_2'] == indicator_2]
                  ['score_max'].values[0]) / A_I[k] / 100
            for indicator_2 in df_tmp['indicator_2']
        }
    A_II[RRDA_AI_indicater_1]['家庭经济困难度指数'] = 0
    A_II[RRDA_AI_indicater_1]['家庭经济困难度指数'] = 1 - sum(
        A_II[RRDA_AI_indicater_1].values())

    return A_I, A_II


def get_AI_AII(df_idx):
    """
    根据指标体系v1获取A_I, A_II
    df_idx: 指标体系v1的DataFrame
    """
    # 按照 indicator_2 分组并获取每组的第一个元素
    df_grouped = df_idx.groupby('indicator_2')
    df_first = df_grouped.first()
    new_df = df_first.reset_index()

    # 删除不需要的列
    new_df.drop(
        columns=['Unnamed: 0', 'indicator_3', 'score', 'normalized_score'],
        inplace=True,
        errors='ignore')

    # 按照 indicator_1 分组并计算每组 score_max 的和
    new_df_grouped = new_df.groupby('indicator_1')
    A_I = new_df_grouped['score_max'].sum().to_dict()

    # 每一项的值都除以 100
    A_I = {k: v / 100 for k, v in A_I.items()}

    # 根据 new_df 获取 A_II
    A_II = {}
    for k in A_I.keys():
        df_tmp = new_df[new_df['indicator_1'] == k]
        A_II[k] = {
            indicator_2:
            float(df_tmp[df_tmp['indicator_2'] == indicator_2]
                  ['score_max'].values[0]) / A_I[k] / 100
            for indicator_2 in df_tmp['indicator_2']
        }

    return A_I, A_II


def extract_indicator_number(indicator_name):
    """
    从指标名称中提取前面的数字编号
    例如: "1.学生户籍所在区县" -> 1
         "21.父亲受教育程度" -> 21
    """
    import re
    match = re.match(r'^(\d+)\.', str(indicator_name))
    if match:
        return int(match.group(1))
    return float('inf')  # 如果没有数字，放到最后


def sort_indicators_by_number(indicators):
    """
    根据指标名称前面的数字对指标列表进行排序
    """
    return sorted(indicators, key=extract_indicator_number)


def get_one_score(indicator_2, indicator_3, df_zbtx):
    '''
    获取某个指标的分数
    '''
    df_tmp = df_zbtx[
        (df_zbtx['indicator_2'] == indicator_2)
        & (df_zbtx['indicator_3'] == indicator_3)]
    if not df_tmp.empty:
        return float(df_tmp['score'].values[0])
    else:
        print(f'Warning: 指标未找到 - {indicator_2} - {indicator_3}')
        return None


def data_clean(df_std, df_zbtx):
    """
    :param df_std: 困难认定数据
    数据清洗规则：
    规则1. "2.受抚养情况"为"单亲家庭子女，父母一方去世，由父母其中一方独自抚养孩子"或"单亲家庭子女，父母离异，由父母其中一方独自抚养孩子"
        时，要么‘3.父亲常住地’、‘6.父亲职业’、‘12.父亲健康状况’、‘21.父亲受教育程度’均为空且‘4.母亲常住地’、‘7.母亲职业’、‘13.母亲健康状况’、‘22.母亲受教育程度’均不为空；
        要么‘4.母亲常住地’、‘7.母亲职业’、‘13.母亲健康状况’、‘22.母亲受教育程度’均为空且‘3.父亲常住地’、‘6.父亲职业’、‘12.父亲健康状况’、‘21.父亲受教育程度’均不为空；
        "2.受抚养情况"为"父母双方均不具有或均未承担抚养责任"时，
        ‘3.父亲常住地’、‘6.父亲职业’、‘12.父亲健康状况’、‘21.父亲受教育程度’、
        ‘4.母亲常住地’、‘7.母亲职业’、‘13.母亲健康状况’、‘22.母亲受教育程度’均为空；
    规则2. "5.家庭劳动力数量"列为"无劳动力（零就业家庭）"时，",6.父亲职业"和"7.母亲职业"必须为"无工作无收入"或为空；
       "1人"时，"6.父亲职业"或"7.母亲职业"不能都选择除了"无工作无收入"以外的选项，可以都是空；
    
       注：须提前判断是否有上述列存在
    """

    father_cols = ['3.父亲常住地', '6.父亲职业', '12.父亲健康状况', '21.父亲受教育程度']
    mother_cols = ['4.母亲常住地', '7.母亲职业', '13.母亲健康状况', '22.母亲受教育程度']
    
    def clean_1(df):
        if '2.受抚养情况' in df.columns:
            cond1 = df['2.受抚养情况'].isin(
                ['单亲家庭子女，父母一方去世，由父母其中一方独自抚养孩子', '单亲家庭子女，父母离异，由父母其中一方独自抚养孩子'])
            cond2 = df['2.受抚养情况'] == '父母双方均不具有或均未承担抚养责任'

            for col in father_cols + mother_cols:
                if col not in df.columns:
                    raise ValueError(f"缺少必要的列: {col}")

            cond1_father_empty = df[father_cols].isnull().all(axis=1)
            cond1_mother_not_empty = df[mother_cols].notnull().any(axis=1)
            cond1_mother_empty = df[mother_cols].isnull().all(axis=1)
            cond1_father_not_empty = df[father_cols].notnull().any(axis=1)

            cond2_all_empty = df[father_cols +
                                    mother_cols].isnull().all(axis=1)

            final_cond = ((cond1 & ((cond1_father_empty & cond1_mother_not_empty) |
                                    (cond1_mother_empty & cond1_father_not_empty)))
                        | (cond2 & cond2_all_empty) | (~cond1 & ~cond2))
        return final_cond

    def clean_2(df):
        if '5.家庭劳动力数量' in df.columns:
            cond_zero = df['5.家庭劳动力数量'] == '无劳动力（零就业家庭）'
            cond_one = df['5.家庭劳动力数量'] == '1人'

            for col in ['6.父亲职业', '7.母亲职业']:
                if col not in df.columns:
                    raise ValueError(f"缺少必要的列: {col}")

            cond_zero_valid = df[cond_zero].apply(
                lambda row: all(row[col] in ['无工作无收入', None, '']
                                for col in ['6.父亲职业', '7.母亲职业']),
                axis=1)
            cond_one_valid = df[cond_one].apply(
                lambda row: not all(row[col] not in ['无工作无收入', None, '']
                                    for col in ['6.父亲职业', '7.母亲职业']),
                axis=1)
            final_cond = ((cond_zero & cond_zero_valid) |
                          (cond_one & cond_one_valid) | (~cond_zero & ~cond_one))
        return final_cond

    # df_std_bad1 = df_std[~clean_1(df_std)]
    # df_std_bad2 = df_std[~clean_2(df_std)]
    # df_std_cleaned_2 = df_std[clean_2(df_std)]
    # df_std_cleaned = df_std[clean_1(df_std) & clean_2(df_std)]

    # return df_std_cleaned, df_std_cleaned_2, df_std_bad1, df_std_bad2

    # 仅考虑clean_1，将df_std中不满足clean_1的行进行改造
    # 具体规则：对于双方均不抚养的，将父母相关信息置为空；对于父母一方抚养的，由于不知道是父母哪一方抚养，因此分如下情况处理：当父母一方信息为空时，保留另一方信息；当父母双方信息均不为空时，分别计算父亲和母亲的4个分数之和，保留低值对应的一方信息，另一方置为空。

    df_std.replace('(空)', None, inplace=True)
    
    # 获取不满足clean_1的行
    df_std_bad1 = df_std[~clean_1(df_std)]
    # 复制一份df_std用于清洗
    df_std_cleaned = df_std.copy()

    # df_zbtx = pd.read_csv(zbtx_path)

    # 对不满足clean_1的行进行处理
    for idx, row in df_std_bad1.iterrows():
        if row['2.受抚养情况'] == '父母双方均不具有或均未承担抚养责任':
            for col in father_cols + mother_cols:
                df_std_cleaned.at[idx, col] = None
        else: # 单亲家庭子女
            father_score = 0
            mother_score = 0
            # 计算父亲的4个分数之和
            for col in father_cols:
                if pd.notnull(row[col]):
                    father_score += get_one_score(col, row[col], df_zbtx)
            # 计算母亲的4个分数之和
            for col in mother_cols:
                if pd.notnull(row[col]):
                    mother_score += get_one_score(col, row[col], df_zbtx)
            # 根据分数大小决定保留哪一方的信息
            if father_score <= mother_score: # 如果父亲分数较低，保留父亲信息
                for col in mother_cols:
                    df_std_cleaned.at[idx, col] = None
            else:
                for col in father_cols:
                    df_std_cleaned.at[idx, col] = None
        df_std_cleaned.at[idx, '未通过规则1筛选'] = True

    return df_std_cleaned

if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings('ignore')
    school_names = [
        '西北工业大学',
        '西安工业大学',
        '西安工程大学',
        '西北政法大学',
        'xagcdx_with_difficulty',
    ]

    # for school_name in school_names:
    #     print(f"处理学校: {school_name}")
    #     df_std = pd.read_csv(f"D:\\Working\\助管\\精准资助\\试点验证方案\\真实结果\\with_diff\\{school_name}.csv")
    #     # 将“(空)”替换为NaN
    #     df_std.replace('(空)', None, inplace=True)
    #     print(f"清洗前数据量: {df_std.shape[0]}")
    #     df_std_cleaned, df_std_cleaned_2, df_std_bad1, df_std_bad2 = data_clean(df_std)
    #     print(f'规则1清洗: {df_std_bad1.shape[0]}')
    #     print(f'规则2清洗: {df_std_bad2.shape[0]}')
    #     print(f"仅规则2清洗后: {df_std_cleaned_2.shape[0]}")
    #     print(f"规则1，2清洗后: {df_std_cleaned.shape[0]}")
    #     # 保存清洗后的数据和剔除的数据
    #     df_std_cleaned.to_csv(f"D:\\Working\\助管\\精准资助\\试点验证方案\\真实结果\\with_diff\\清洗测试\\{school_name}_cleaned.csv", index=False, encoding='utf_8_sig')
    #     df_std_bad1.to_csv(f"D:\\Working\\助管\\精准资助\\试点验证方案\\真实结果\\with_diff\\清洗测试\\{school_name}_bad1.csv", index=False, encoding='utf_8_sig')
    #     df_std_bad2.to_csv(f"D:\\Working\\助管\\精准资助\\试点验证方案\\真实结果\\with_diff\\清洗测试\\{school_name}_bad2.csv", index=False, encoding='utf_8_sig')
    #     print()

    df_std = pd.read_csv(f"D:\\Working\\助管\\精准资助\\试点验证方案\\真实结果\\with_diff\\{school_names[0]}.csv")
    df_zbtx = pd.read_csv('tmp\\整合版量化指标体系_20260115(最新).csv')

    df_clean = data_clean(df_std, df_zbtx)
    df_clean.to_csv(f'D:\\Working\\助管\\精准资助\\试点验证方案\\真实结果\\with_diff\\清洗测试\\{school_names[-1]}_cleaned_v2.csv', index=False, encoding='utf_8_sig')
    # TODO 规则1与规则2分开，规则1不删除数据

    # 处理学校: 西北工业大学
    # 清洗前数据行数: 449
    # 规则1和规则2剔除的行数: 88
    # 清洗后数据行数: 361

    # 处理学校: 西安工业大学
    # 清洗前数据行数: 4781
    # 规则1和规则2剔除的行数: 820
    # 清洗后数据行数: 3961

    # 处理学校: 西安工程大学
    # 清洗前数据行数: 3931
    # 规则1和规则2剔除的行数: 705
    # 清洗后数据行数: 3226

    # 处理学校: 西北政法大学
    # 清洗前数据行数: 1908
    # 规则1和规则2剔除的行数: 357
    # 清洗后数据行数: 1551