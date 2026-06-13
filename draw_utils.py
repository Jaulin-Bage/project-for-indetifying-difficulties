import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import io

DEFAULT_DPI = 250
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局颜色列表 - 用于权重变化曲线
GLOBAL_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
    '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
]

def visualization(df: pd.DataFrame, V: list, filepath=None):
    pl = []
    for res in df['实际认定结果'].tolist():
        lev = 0
        if res == V[3]:
            lev = 0
        elif res == V[2]:
            lev = 1
        elif res == V[1]:
            lev = 2
        elif res == V[0]:
            lev = 3
        else:
            print('结果有误！')
            return
        pl.append(lev)

    plt.figure(figsize=(8, 3), dpi=DEFAULT_DPI)
    plt.scatter(x=df['困难度分数'].tolist(), y=pl, alpha=0.3)
    plt.yticks([0, 1, 2, 3], ['不困难', '一般困难', '困难', '特别困难'])
    plt.xlabel('困难度分值')
    plt.ylabel('真实困难程度')
    plt.title('困难度分值与真实困难程度散点图')
    plt.grid(True)
    sns.despine(ax=plt.gca())

    # save the plot
    if filepath is not None:
        plt.savefig(filepath)

    # plt.show()


# 输入两个列表，将列表b分为4个列表
def split_list(a, b):
    result = {}
    for x, y in zip(a, b):
        if x not in result:
            result[x] = []
        result[x].append(y)
    print(result.keys())
    return result


def violin_plot(a, b, filepath=None):
    result = split_list(a, b)
    data = []
    data.append(result['特别困难'])
    data.append(result['困难'])
    data.append(result['一般困难'])
    data.append(result['不困难'])

    plt.figure(figsize=(6, 6), dpi=DEFAULT_DPI)
    sns.set_style('whitegrid')
    sns.violinplot(data=data, palette='Set2')
    plt.xticks([0, 1, 2, 3], ['Lv.1', 'Lv.2', 'Lv.3', 'Others'])
    plt.ylabel('Poverty Scores')
    plt.xlabel('Poverty Levels')
    sns.despine()

    if filepath is not None:
        plt.savefig(filepath)
    # plt.savefig('./data/result/困难度分值分布小提琴图.png')
    # plt.show()


def kde_plot(a, b, V, filepath=None):

    result = split_list(a, b)
    data = [
        result['特别困难'],
        result['困难'],
        result['一般困难'],
        result['不困难']
    ]
    
    # 增加图形高度
    plt.figure(figsize=(6, 6), dpi=DEFAULT_DPI)  # 从6改为8

    for i in range(4):
        plt.subplot(4, 1, i + 1)
        sns.kdeplot(data[i], fill=True)
        plt.xlim(0, 100)
        if i < 3:
            plt.xticks([])
        plt.ylabel('')
        plt.title(V[i])
        plt.grid(False)
        sns.despine(ax=plt.gca())
    
    plt.tight_layout()
    # 增加底部间距
    plt.subplots_adjust(bottom=0.08)  # 调整这个值直到合适
    plt.xlabel('困难度分值', size=9)

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')  # 添加bbox_inches参数

def draw_bar(a, b, V, filepath=None, rotation=0):
    '''统计4种困难程度的数量'''

    result = split_list(a, b)
    data = [
        result['特别困难'],
        result['困难'],
        result['一般困难'],
        result['不困难']
    ]

    # 调整画布大小（宽度增加，防止倾斜标签被截断）
    plt.figure(figsize=(6, 6), dpi=DEFAULT_DPI)
    bars = plt.bar(V, [len(data[i]) for i in range(4)])
    
    plt.xlabel('困难程度')
    plt.ylabel('数量')
    plt.title('各困难度人数统计')
    plt.xticks(rotation=rotation, ha='right')  # ha='right' 让标签右对齐，避免重叠
    sns.despine()

    # 自动调整布局 + 手动增加底部空间
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 增加底部边距

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')  # 关键：防止裁剪


def draw_matrix(df, col1, col2, filepath=None):

    # Create a confusion matrix
    confusion_matrix = pd.crosstab(df[col1], df[col2], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6), dpi=DEFAULT_DPI)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('人工认定结果与算法认定结果一致性分析')
    plt.xlabel('算法认定结果')
    plt.ylabel('人工认定结果')
    sns.despine()

    if filepath:
        plt.savefig(filepath)
    # else:
        # plt.show()



def draw_plot(data, title, xlabel, ylabel, file=None, work_dir=None):
    """ 绘制损失曲线 """
    fig = plt.figure(figsize=(5, 5), dpi=DEFAULT_DPI)
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=15)
    sns.despine()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_data = buf.read()
    plt.close(fig)
    return img_data


def draw_plots(train_accs, test_accs, title, xlabel, ylabel, file=None, work_dir=None):
    """ 绘制训练集和测试集准确率曲线 """
    fig = plt.figure(figsize=(5, 5), dpi=DEFAULT_DPI)
    plt.plot(train_accs, label='训练集准确率')
    plt.plot(test_accs, label='测试集准确率')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=15)
    plt.legend()
    sns.despine()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_data = buf.read()
    plt.close(fig)
    return img_data


def draw_all_weights_overlay(normalized_weights_list, A_II_labels, work_dir=None):
    """ 绘制所有权重变化曲线叠加图（不显示legend） """
    normalized_weights_list = np.array(normalized_weights_list)
    num_indicators = normalized_weights_list.shape[1]
    
    fig = plt.figure(figsize=(5, 5), dpi=DEFAULT_DPI)
    
    # 绘制所有权重曲线，使用全局颜色列表
    for i in range(num_indicators):
        color = GLOBAL_COLORS[i % len(GLOBAL_COLORS)]
        plt.plot(normalized_weights_list[:, i], color=color, alpha=0.8, linewidth=1.5)
    
    plt.xlabel('训练轮次')
    plt.ylabel('权重值')
    plt.title('所有指标权重变化', fontsize=15)
    plt.grid(True, alpha=0.3)
    sns.despine()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_data = buf.read()
    plt.close(fig)
    
    # 返回图像数据和颜色映射
    color_mapping = {A_II_labels[i]: GLOBAL_COLORS[i % len(GLOBAL_COLORS)] 
                     for i in range(num_indicators)}
    
    return img_data, color_mapping


def draw_weights(normalized_weights_list, init_weights, A_II_labels, work_dir=None):
    """ 绘制权重变化 """
    normalized_weights_list = np.array(normalized_weights_list)
    
    # 将init_weights作为第0轮权重插入到列表开头
    init_weights_array = np.array(init_weights).reshape(1, -1)
    normalized_weights_list = np.vstack([init_weights_array, normalized_weights_list])
    
    num_indicators = normalized_weights_list.shape[1]
    
    # 计算所有指标中最大的Y轴范围（变化量）
    max_range = 0
    for i in range(num_indicators):
        data_min = np.min(normalized_weights_list[:, i])
        data_max = np.max(normalized_weights_list[:, i])
        data_range = data_max - data_min
        if data_range > max_range:
            max_range = data_range
    
    # 添加边距（10%）
    y_range = max_range * 1.1
    
    # 计算需要的行数（向上取整）
    nrows = (num_indicators + 3) // 4  # 每行4个子图
    ncols = 4
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15), dpi=DEFAULT_DPI)
    
    # 确保axes是二维数组（即使只有一行）
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    # 绘制每个指标的权重变化
    for i in range(num_indicators):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 使用全局颜色列表中的颜色
        color = GLOBAL_COLORS[i % len(GLOBAL_COLORS)]
        ax.plot(normalized_weights_list[:, i], color=color, linewidth=3)
        
        # 标记第0轮的权重值
        initial_weight = normalized_weights_list[0, i]
        ax.plot(0, initial_weight, 'o', color=color, markersize=6, zorder=5)
        ax.text(0, initial_weight, f'{initial_weight:.2f}', 
                fontsize=16, ha='left', va='bottom', fontweight='bold')
        
        # 标记最后一轮的权重值
        final_weight = normalized_weights_list[-1, i]
        final_epoch = len(normalized_weights_list) - 1
        ax.plot(final_epoch, final_weight, 'o', color=color, markersize=6, zorder=5)
        ax.text(final_epoch, final_weight, f'{final_weight:.4f}', 
                fontsize=16, ha='left', va='bottom', fontweight='bold')
        
        title = A_II_labels[i] if len(
            A_II_labels[i]
        ) <= 10 else A_II_labels[i][:12] + '...'
        ax.set_title(title, fontsize=15)
        
        # 计算当前指标的数据范围中心点
        data_min = np.min(normalized_weights_list[:, i])
        data_max = np.max(normalized_weights_list[:, i])
        data_center = (data_min + data_max) / 2
        
        # 使用统一的y轴范围，但以当前数据的中心点为基准
        y_min = data_center - y_range / 2
        y_max = data_center + y_range / 2
        ax.set_ylim(y_min, y_max)
        
        ax.grid(True)
        sns.despine(ax=ax)
    
    # 隐藏最后一行多余的子图
    total_plots = nrows * ncols
    for i in range(num_indicators, total_plots):
        row = i // 4
        col = i % 4
        axes[row, col].axis('off')  # 隐藏多余的子图

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_data = buf.read()
    plt.close(fig)
    return img_data