"""
Utility functions for data analysis.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from params import V

PLOT_FONT_SIZE = 16


def get_available_schools_and_years(dir="output", file_suffix="_AII.csv"):
    '''获取可用的学校和年份列表'''
    schools = []
    years = set()
    
    if not os.path.exists(dir):
        return schools, sorted(years)
        
    for school in os.listdir(dir):
        school_path = os.path.join(dir, school)
        if os.path.isdir(school_path):
            schools.append(school)
            for file in os.listdir(school_path):
                if file.endswith(file_suffix):
                    year = file[:-len(file_suffix)]
                    years.add(year)
    
    return sorted(schools), sorted(years)


def get_csv_files(dir):
    '''获取指定目录及其子目录下的所有CSV文件路径。'''
    csv_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def get_sum_students(dir):
    '''获取指定目录下所有CSV文件中学生的总人数。'''
    csv_files = get_csv_files(dir)
    total_students = 0
    for file in csv_files:
        df = pd.read_csv(file)
        total_students += df.shape[0]
    return total_students


def get_selected_info(dir,
                      include_schools=None,
                      include_years=None,
                      file_suffix="_AII.csv"):
    '''获取参与认定的概览信息
    :param dir: 目录路径，包含各学校子目录，每个子目录包含各年份的CSV文件
    :param include_schools: 包含的学校列表，None表示包含所有学校
    :param include_years: 包含的年份列表，None表示包含所有年份
    :return: 包含概览信息的字典
    返回结构:
    {
      "学校1": {
          "年份1": {
                "学生总数": int,
                "困难程度": {'特别困难': int, '困难': int, '一般困难': int, '不困难': int},
                "困难度分数中位数": float,
                "平均困难度分数": float,
                "最大困难度分数": float,
                "最小困难度分数": float,
          },
          "年份2": {
                "学生总数": int,
                ...
          },
          ...
        },
        "学校2": {
            ...
        },
        ...
    }
    '''
    res = {}
    info = {}

    for school in os.listdir(dir):
        if include_schools and school not in include_schools:
            continue  # 跳过不包含的学校
        school_path = os.path.join(dir, school)
        if os.path.isdir(school_path):
            info[school] = {}
            for file in os.listdir(school_path):
                if file.endswith(file_suffix):
                    year = file[:-len(file_suffix)]  # 去掉.csv
                    if include_years and year not in include_years:
                        continue  # 跳过不包含的年份
                    file_path = os.path.join(school_path, file)
                    df = pd.read_csv(file_path)
                    info[school][year] = {
                        "学生总数": df.shape[0],
                        "困难程度": df["算法认定结果"].value_counts().to_dict(),
                        "困难度分数中位数":
                        np.median(df["困难度分数"]) if df.shape[0] > 0 else 0,
                        "平均困难度分数":
                        np.mean(df["困难度分数"]) if df.shape[0] > 0 else 0,
                        "最大困难度分数":
                        np.max(df["困难度分数"]) if df.shape[0] > 0 else 0,
                        "最小困难度分数":
                        np.min(df["困难度分数"]) if df.shape[0] > 0 else 0,
                    }
    res['info'] = info
    res['total_students'] = sum(info[school][year]["学生总数"] for school in info
                                for year in info[school])
    res['total_schools'] = len(info)
    res['total_years'] = len(
        set(year for school in info for year in info[school]))
    return res


def plot_people_school(info):
    '''绘制参与认定的人员数量统计图表，x轴为学校，y轴为人数'''
    # 读取数据
    schools = list(info.keys())
    total_students = [
        sum(year_info["学生总数"] for year_info in info[school].values())
        for school in schools
    ]
    # 绘图
    fig = plt.figure(figsize=(1 + len(schools), 4))
    plt.bar(schools, total_students, color='skyblue')
    plt.xlabel('学校', fontsize=PLOT_FONT_SIZE)
    plt.ylabel('人数', fontsize=PLOT_FONT_SIZE)
    plt.title('各学校参与认定的人员数量统计', fontsize=PLOT_FONT_SIZE)
    plt.xticks(rotation=45, fontsize=PLOT_FONT_SIZE)
    plt.yticks(fontsize=PLOT_FONT_SIZE)
    plt.tight_layout()
    return fig


def plot_people_year(info):
    '''绘制参与认定的人员数量统计图表，x轴为年份，y轴为人数'''
    # 读取数据
    year_dict = {}
    for school in info:
        for year in info[school]:
            if year not in year_dict:
                year_dict[year] = 0
            year_dict[year] += info[school][year]["学生总数"]
    years = sorted(year_dict.keys())
    total_students = [year_dict[year] for year in years]
    # 绘图
    fig = plt.figure(figsize=(1 + len(years), 4))
    plt.bar(years, total_students, color='skyblue')
    plt.xlabel('年份', fontsize=PLOT_FONT_SIZE)
    plt.ylabel('人数', fontsize=PLOT_FONT_SIZE)
    plt.title('各年份参与认定的人员数量统计', fontsize=PLOT_FONT_SIZE)
    plt.xticks(rotation=0, fontsize=PLOT_FONT_SIZE)
    plt.yticks(fontsize=PLOT_FONT_SIZE)
    plt.tight_layout()
    return fig


def plot_poverty_distribution(info):
    '''绘制困难程度分布扇形图'''
    # 读取数据
    poverty_dict = {key: 0 for key in V}
    for school in info:
        for year in info[school]:
            for level, count in info[school][year]["困难程度"].items():
                if level in poverty_dict:
                    poverty_dict[level] += count
    levels = list(poverty_dict.keys())
    counts = [poverty_dict[level] for level in levels]
    # 绘图
    color_palette = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(counts,
                                      labels=levels,
                                      autopct='%1.1f%%',
                                      startangle=0,
                                      colors=color_palette)
    for text in texts:
        text.set_fontsize(PLOT_FONT_SIZE)
    for autotext in autotexts:
        autotext.set_fontsize(PLOT_FONT_SIZE)
    ax.set_title('困难程度分布扇形图', fontsize=PLOT_FONT_SIZE)
    plt.axis('equal')  # 使饼图为圆形
    return fig


def plot_poverty_trend(info):
    """绘制困难程度随年份变化的趋势折线图, x轴为年份, 折线为平均、最大、最小困难度分数、中位数"""
    # 读取数据
    year_dict = {}
    for school in info:
        for year in info[school]:
            if year not in year_dict:
                year_dict[year] = {
                    "总人数": 0,
                    "总分数": 0.0,
                    "最大分数": float('-inf'),
                    "最小分数": float('inf'),
                    "所有分数": [],
                }
            year_info = info[school][year]
            year_dict[year]["总人数"] += year_info["学生总数"]
            year_dict[year]["总分数"] += year_info["平均困难度分数"] * year_info["学生总数"]
            year_dict[year]["最大分数"] = max(year_dict[year]["最大分数"],
                                          year_info["最大困难度分数"])
            year_dict[year]["最小分数"] = min(year_dict[year]["最小分数"],
                                          year_info["最小困难度分数"])
            year_dict[year]["所有分数"].append(year_info["困难度分数中位数"])

    years = sorted(year_dict.keys())
    avg_scores = [(year_dict[year]["总分数"] /
                   year_dict[year]["总人数"]) if year_dict[year]["总人数"] > 0 else 0
                  for year in years]
    max_scores = [year_dict[year]["最大分数"] for year in years]
    min_scores = [
        year_dict[year]["最小分数"]
        if year_dict[year]["最小分数"] != float('inf') else 0 for year in years
    ]
    median_scores = [
        np.median(year_dict[year]["所有分数"])
        if len(year_dict[year]["所有分数"]) > 0 else 0 for year in years
    ]

    # 绘图
    fig = plt.figure(figsize=(1 + len(years), 4))
    plt.plot(years, avg_scores, marker='o', label='平均困难度分数')
    plt.plot(years, max_scores, marker='o', label='最大困难度分数')
    plt.plot(years, min_scores, marker='o', label='最小困难度分数')
    plt.plot(years, median_scores, marker='o', label='中位数困难度分数')
    plt.xlabel('年份', fontsize=PLOT_FONT_SIZE)
    plt.ylabel('困难度分数', fontsize=PLOT_FONT_SIZE)
    plt.title('困难度分数随年份变化的趋势', fontsize=PLOT_FONT_SIZE)
    plt.xticks(rotation=0, fontsize=PLOT_FONT_SIZE)
    plt.yticks(fontsize=PLOT_FONT_SIZE)
    plt.legend(fontsize=PLOT_FONT_SIZE)
    plt.tight_layout()
    return fig


def plot_poverty_school_dist(info):
    """绘制各学校困难程度分布的条形图，x轴为学校，y轴为困难度分数，展示平均、最大、最小困难度分数、中位数柱子"""
    # 读取数据
    schools = list(info.keys())
    avg_scores = []
    max_scores = []
    min_scores = []
    median_scores = []

    for school in schools:
        # 计算平均困难度分数（按学生人数加权）
        total_students = sum(year_info["学生总数"]
                             for year_info in info[school].values())
        total_score = sum(year_info["平均困难度分数"] * year_info["学生总数"]
                          for year_info in info[school].values())
        avg_scores.append((total_score /
                           total_students) if total_students > 0 else 0)

        # 计算最大困难度分数
        max_scores.append(
            max(year_info["最大困难度分数"] for year_info in info[school].values()))

        # 计算最小困难度分数（处理inf值）
        min_values = [
            year_info["最小困难度分数"] for year_info in info[school].values()
            if year_info["最小困难度分数"] != float('inf')
        ]
        min_scores.append(min(min_values) if min_values else 0)

        # 计算中位数困难度分数
        median_values = [
            year_info["困难度分数中位数"] for year_info in info[school].values()
        ]
        median_scores.append(np.median(median_values) if median_values else 0)

    # 设置柱子位置和宽度
    x = np.arange(len(schools))
    width = 0.18  # 柱子宽度，稍微减小以避免重叠

    # 定义颜色方案
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # 蓝色、红色、绿色、橙色

    # 绘图
    fig_width = max(8, 2 + len(schools) * 0.8)  # 根据学校数量调整图形宽度
    fig = plt.figure(figsize=(fig_width, 6))

    # 绘制四组柱子，使用不同颜色
    bars1 = plt.bar(x - 1.5 * width,
                    avg_scores,
                    width,
                    label='平均',
                    color=colors[0],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5)
    bars2 = plt.bar(x - 0.5 * width,
                    max_scores,
                    width,
                    label='最大',
                    color=colors[1],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5)
    bars3 = plt.bar(x + 0.5 * width,
                    min_scores,
                    width,
                    label='最小',
                    color=colors[2],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5)
    bars4 = plt.bar(x + 1.5 * width,
                    median_scores,
                    width,
                    label='中位数',
                    color=colors[3],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5)

    # 在柱子上方添加数值标签
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     height + 0.5,
                     f'{value:.1f}',
                     ha='center',
                     va='bottom',
                     fontsize=PLOT_FONT_SIZE)

    add_value_labels(bars1, avg_scores)
    add_value_labels(bars2, max_scores)
    add_value_labels(bars3, min_scores)
    add_value_labels(bars4, median_scores)

    # 设置坐标轴和标题
    plt.xlabel('学校', fontsize=PLOT_FONT_SIZE)
    plt.ylabel('困难度分数', fontsize=PLOT_FONT_SIZE)
    plt.ylim(0, 100)
    plt.title('各学校困难度分数统计对比', fontsize=PLOT_FONT_SIZE)
    plt.xticks(x, schools, rotation=45, ha='right', fontsize=PLOT_FONT_SIZE)
    plt.yticks(fontsize=PLOT_FONT_SIZE)

    # 添加图例
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=PLOT_FONT_SIZE)

    # 添加网格线
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 调整布局
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    directory = "output"
    include_schools = None  # e.g., ['SchoolA', 'SchoolB']
    include_years = None  # e.g., ['2021', '2022']
    res = get_selected_info(directory, include_schools, include_years)
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(res)
    info = res['info']
    # plot_poverty_distribution(info)
    # plot_poverty_trend(info)
    # plot_poverty_school_dist(info)
    # plot_people_school(info)
    # plot_people_year(info)
    # plot_poverty_distribution(info)
    # plot_poverty_trend(info)
    plot_poverty_school_dist(info)
