import pandas as pd
import numpy as np


def calc_gama(w_i, w_sh, alpha):
    '''计算家庭经济困难度指数'''
    gama = alpha * w_i + (1 - alpha) * w_sh
    return gama


def calc_w_i(std_avg_income,
             hometown_avg_income,
             hometown_lowest_income,
             alpha=1):
    '''计算家庭相对所在地经济水平指数'''
    w_i = 0
    if std_avg_income <= hometown_lowest_income:
        w_i = alpha
    elif std_avg_income <= hometown_avg_income:
        w_i = alpha * ((hometown_avg_income - std_avg_income) /
                       (hometown_avg_income - hometown_lowest_income))
    else:
        w_i = 0
    assert w_i <= alpha and w_i >= 0, 'w_i计算有误!'
    return w_i


def calc_w_sh(school_economic, hometown_economic, gdp=85698):
    '''计算家校经济水平差指数'''
    w_sh = (school_economic - hometown_economic) / gdp
    return w_sh


def calc_w_sh_norm(w_sh, w_sh_min, w_sh_max):
    '''计算正则化家校经济水平差指数'''
    w_sh_norm = (w_sh - w_sh_min) / (w_sh_max - w_sh_min)
    assert w_sh_norm <= 1 and w_sh_norm >= 0, 'w_sh计算有误!'
    return w_sh_norm


def load_data(city_income_data, std_data):
    # 加载数据
    df_city_income = pd.read_csv(city_income_data)
    df_city_income.dropna(how='all')

    df_std = pd.read_csv(std_data)
    df_std.dropna(how='all')

    # 去除未收集数据的项
    cities = df_city_income['市'].tolist()
    df_std = df_std[df_std['家庭所在地地址'].str.contains('|'.join(cities), na=False)]

    return df_city_income, df_std


def data_clean(df_std: pd.DataFrame):
    # 数据真实性检验

    fake_list = []
    for i in [1111, 11111, 111111, 1111111]:
        for j in range(1, 10):
            fake_list.append(i * j)

    zero_score = (df_std['学生家庭年收入（元/年）'].isin(fake_list)) | (
        df_std['学生家庭年收入（元/年）'] < 0)
    try:
        zero_score = zero_score | (df_std['举报提供虚假信息证明材料'] == '有')
    except:
        pass
    scores = np.array([-1.] * df_std.shape[0])
    scores[zero_score] = 0.

    return scores


def calc_constant(df_city_income):
    # 计算相关常数
    df_city_income.loc[:, '城镇经济水平'] = df_city_income[
        '城镇人均可支配收入'] - df_city_income['城镇最低收入']
    df_city_income.loc[:, '农村经济水平'] = df_city_income[
        '农村人均可支配收入'] - df_city_income['农村最低收入']

    eco_town = df_city_income['城镇经济水平'].tolist()
    eco_vill = df_city_income['农村经济水平'].tolist()
    eco_all = eco_town + eco_vill

    w_sh_min, w_sh_max = 1e9, -1e9
    for eco1 in eco_town:
        for eco2 in eco_all:
            w_sh_min = min(calc_w_sh(eco1, eco2), w_sh_min)
            w_sh_max = max(calc_w_sh(eco1, eco2), w_sh_max)

    return w_sh_min, w_sh_max


def evaluate(df_std: pd.DataFrame):
    pov_levels = ['特别困难', '困难', '一般困难', '不困难']
    poverty_res = df_std['实际认定结果'].tolist()

    # 准确率计算
    level_nums = [poverty_res.count(pov_level) for pov_level in pov_levels]
    excepted_res = []
    for pov_level, level_num in zip(pov_levels, level_nums):
        excepted_res += [pov_level] * level_num

    df_std_sorted = df_std.sort_values(by='家庭经济困难度指数', ascending=False)
    df_std_sorted['算法认定结果'] = excepted_res

    acc = df_std_sorted[
        df_std_sorted['算法认定结果'] ==
        df_std_sorted['实际认定结果']].shape[0] / df_std_sorted.shape[0]
    print(f'RRDA算法准确率：{acc * 100}%')

    return acc, df_std_sorted


def determination(df_std: pd.DataFrame, poverty_perc: dict):
    '''自定义各个困难度比重'''
    std_nums = df_std.shape[0]

    p1 = int(
        std_nums * poverty_perc['一般困难'] /
        (poverty_perc['一般困难'] + poverty_perc['困难'] + poverty_perc['特别困难']))
    p2 = int(
        std_nums * poverty_perc['困难'] /
        (poverty_perc['一般困难'] + poverty_perc['困难'] + poverty_perc['特别困难']))
    p3 = int(
        std_nums * poverty_perc['特别困难'] /
        (poverty_perc['一般困难'] + poverty_perc['困难'] + poverty_perc['特别困难']))
    p0 = std_nums - p1 - p2 - p3

    expected_hard_list = ['特别困难'] * p1 + \
        ['困难'] * p2 + ['一般困难'] * p3 + ['不困难'] * p0
    df_std_sorted = df_std.sort_values(by='gama', ascending=False)
    df_std_sorted['expected_res'] = expected_hard_list

    return df_std_sorted


def get_hometown_avg_lowest_income(df_std: pd.DataFrame,
                                   df_city_income: pd.DataFrame):
    city_income = df_city_income.set_index('市')
    hometown_avg_income = []
    hometown_lowest_income = []

    for idx, row in df_std.iterrows():
        city = next((city for city in city_income.index
                     if city in row['家庭所在地地址']), None)
        if city is None:
            avg_income = 0
            lowest_income = 0
        else:
            if row['农村/城市'] == '农村':
                avg_income = city_income.loc[city, '农村人均可支配收入']
                lowest_income = city_income.loc[city, '农村最低收入']
            else:
                avg_income = city_income.loc[city, '城镇人均可支配收入']
                lowest_income = city_income.loc[city, '城镇最低收入']

        hometown_avg_income.append(avg_income)
        hometown_lowest_income.append(lowest_income)

    df_std['家庭所在地平均收入'] = hometown_avg_income
    df_std['家庭所在地最低收入'] = hometown_lowest_income

    return df_std


def get_school_avg_lowest_income(df_std: pd.DataFrame,
                                 df_city_income: pd.DataFrame):
    city_avg_income = df_city_income.set_index('市')['城镇人均可支配收入'].to_dict()
    city_lowest_income = df_city_income.set_index('市')['城镇最低收入'].to_dict()
    school_avg_income = []
    school_lowest_income = []

    for idx, row in df_std.iterrows():
        city = next((city for city in city_avg_income if city in row['学校所在地']), None)
        if city is None:
            avg_income = 0
            lowest_income = 0
        else:
            avg_income = city_avg_income[city]
            lowest_income = city_lowest_income[city]

        school_avg_income.append(avg_income)
        school_lowest_income.append(lowest_income)

    df_std['学校所在地平均收入'] = school_avg_income
    df_std['学校所在地最低收入'] = school_lowest_income

    return df_std


def RRDA(df_city_income, df_std, alpha) -> pd.DataFrame:

    w_sh_min, w_sh_max = calc_constant(df_city_income)
    print(w_sh_min, w_sh_max)

    scores = data_clean(df_std).tolist()

    # 计算家庭人均年收入
    df_std['学生家庭人均年收入'] = df_std['学生家庭年收入（元/年）'] / df_std['家庭人口数']

    # 添加家庭所在地人均收入、最低收入
    df_std = get_hometown_avg_lowest_income(df_std, df_city_income)
    # 添加学校所在地人均收入、最低收入
    df_std = get_school_avg_lowest_income(df_std, df_city_income)

    w_is, w_sh_norms, gamas = [], [], []
    
    for i, (_, row) in enumerate(df_std.iterrows()):
        if i % 500 == 499:
            print(f"RRDA - {i + 1}/{len(df_std)}")
        if scores[i] == 0:
            w_is.append(0.)
            w_sh_norms.append(0.)
            gamas.append(0.)
        else:
            w_i = calc_w_i(std_avg_income=row['学生家庭人均年收入'],
                           hometown_avg_income=row['家庭所在地平均收入'],
                           hometown_lowest_income=row['家庭所在地最低收入'])
            w_sh = calc_w_sh(
                school_economic=row['学校所在地平均收入'] - row['学校所在地最低收入'],
                hometown_economic=row['家庭所在地平均收入'] - row['家庭所在地最低收入'])
            w_sh_norm = calc_w_sh_norm(w_sh, w_sh_min, w_sh_max)
            gama = calc_gama(w_i, w_sh_norm, alpha)

            w_is.append(w_i)
            w_sh_norms.append(w_sh_norm)
            gamas.append(gama)

    df_std['家庭经济相对所在地经济水平指数'] = w_is
    df_std['家校经济水平差指数'] = w_sh_norms
    df_std['家庭经济困难度指数'] = gamas
    
    print("RRDA done.")

    return df_std


def run_RRDA(all_item_std_file, city_income_data='./data/全国各城市人均收入&最低收入.csv'):
    df_city_income, df_std = load_data(city_income_data=city_income_data,
                                            std_data=all_item_std_file)

    df_std = RRDA(df_city_income, df_std, alpha=0.279)
    print("RRDA calculation done.")

    return df_std
    