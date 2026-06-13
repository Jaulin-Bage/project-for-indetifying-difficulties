import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体（微软雅黑）
font_path = r"C:\Windows\Fonts\msyh.ttc"  # Windows下微软雅黑路径
font_prop = FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取数据
file_path = '西工大资助数据+地址（22+23+24）.xlsx'  # 修改为你的实际路径
df = pd.read_excel(file_path)
df.dropna(how='all', inplace=True)

# 设置Seaborn主题
sns.set_theme(style='whitegrid')

# -------------------- 分类字段定义 --------------------

# 定义低保户类别（只要“家庭困难类型”中包含“低保”即为低保户）
def classify_low_income(text):
    if pd.isna(text):
        return '未知'
    if '低保' in str(text):
        return '低保户'
    return '非低保户'

df['低保户类别'] = df['家庭困难类型'].apply(classify_low_income)

# -------------------- 图1：评定困难类型分布 --------------------
plt.figure(figsize=(10,6))
counts = df['评定困难类型'].value_counts()
ax1 = sns.barplot(x=counts.index, y=counts.values, palette='Reds')
ax1.set_title('学生评定困难类型分布', fontproperties=font_prop, fontsize=18, fontweight='bold', color='#a63603')
ax1.set_xlabel('评定困难类型', fontproperties=font_prop, fontsize=14)
ax1.set_ylabel('学生数量', fontproperties=font_prop, fontsize=14)
ax1.set_xticklabels(ax1.get_xticklabels(), fontproperties=font_prop, fontsize=12, rotation=30)
for p in ax1.patches:
    height = p.get_height()
    ax1.annotate(f'{height}', (p.get_x() + p.get_width()/2, height), ha='center', va='bottom',
                 fontsize=12, fontproperties=font_prop, color='#7f2704')
plt.tight_layout()
plt.show()

# -------------------- 图2：低保户与非低保户学生数量对比 --------------------
# 去除未知值
df_valid_low_income = df[df['低保户类别'].isin(['低保户', '非低保户'])]

plt.figure(figsize=(7,5))
ax2 = sns.countplot(data=df_valid_low_income, x='低保户类别',
                    order=['低保户','非低保户'],
                    palette=['#4c9f70', '#7ea9c7'])
ax2.set_title('低保户与非低保户学生数量对比', fontproperties=font_prop, fontsize=16, fontweight='bold', color='#2b554f')
ax2.set_xlabel('类别', fontproperties=font_prop, fontsize=14)
ax2.set_ylabel('学生数量', fontproperties=font_prop, fontsize=14)
ax2.set_xticklabels(ax2.get_xticklabels(), fontproperties=font_prop, fontsize=13)
for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(f'{height}', (p.get_x() + p.get_width()/2, height), ha='center', va='bottom',
                 fontsize=12, fontproperties=font_prop, color='#2b554f')
plt.tight_layout()
plt.show()

# -------------------- 图3：家庭人均年收入箱线图（培养层次） --------------------
plt.figure(figsize=(8,5))
ax3 = sns.boxplot(data=df, x='培养层次', y='家庭人均年收入', palette='Blues')
ax3.set_title('不同培养层次家庭人均年收入分布', fontproperties=font_prop, fontsize=16, fontweight='bold', color='#2b554f')
ax3.set_xlabel('培养层次', fontproperties=font_prop, fontsize=14)
ax3.set_ylabel('家庭人均年收入（元）', fontproperties=font_prop, fontsize=14)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, fontproperties=font_prop, fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------- 图4：家庭困难类型分布（前10） --------------------
top10 = df['家庭困难类型'].value_counts().head(10)
plt.figure(figsize=(9,6))
ax4 = sns.barplot(y=top10.index, x=top10.values, palette='Blues_r')
ax4.set_title('家庭困难类型分布（前10）', fontproperties=font_prop, fontsize=16, fontweight='bold', color='#2b554f')
ax4.set_xlabel('学生数量', fontproperties=font_prop, fontsize=14)
ax4.set_ylabel('家庭困难类型', fontproperties=font_prop, fontsize=14)
ax4.set_yticklabels(ax4.get_yticklabels(), fontproperties=font_prop, fontsize=13)
for i, v in enumerate(top10.values):
    ax4.text(v + max(top10.values)*0.01, i, str(v),
             fontproperties=font_prop, fontsize=12, color='#2b554f', va='center')
plt.tight_layout()
plt.show()

# # -------------------- 图5：培养层次与现在年级交叉柱状图 --------------------
# plt.figure(figsize=(12,6))
# cross_tab = pd.crosstab(df['培养层次'], df['现在年级'])
# cross_tab = cross_tab.loc[cross_tab.sum(axis=1).sort_values(ascending=False).index]
# ax5 = cross_tab.plot(kind='bar', stacked=False, colormap='Blues', edgecolor='black', figsize=(12,6))
# plt.title('培养层次与现在年级分布', fontproperties=font_prop, fontsize=18, fontweight='bold', color='#2b554f')
# plt.xlabel('培养层次', fontproperties=font_prop, fontsize=15)
# plt.ylabel('学生数量', fontproperties=font_prop, fontsize=15)
# plt.xticks(rotation=30)
# plt.legend(title='现在年级', prop=font_prop, bbox_to_anchor=(1.05,1), loc='upper left', fontsize=12)
# plt.tight_layout()
# plt.show()
# -------------------- 图5：培养层次与现在年级交叉柱状图 --------------------
plt.figure(figsize=(12,6))

# 构建交叉表并按总数排序
cross_tab = pd.crosstab(df['培养层次'], df['现在年级'])
cross_tab = cross_tab.loc[cross_tab.sum(axis=1).sort_values(ascending=False).index]

# 画图
ax5 = cross_tab.plot(kind='bar', stacked=False, colormap='Blues', edgecolor='black', figsize=(12,6))

# 设置标题和坐标轴标题
plt.title('培养层次与现在年级分布', fontproperties=font_prop, fontsize=18, fontweight='bold', color='#2b554f')
plt.xlabel('培养层次', fontproperties=font_prop, fontsize=15)
plt.ylabel('学生数量', fontproperties=font_prop, fontsize=15)

# 设置 X 轴刻度标签字体
for label in ax5.get_xticklabels():
    label.set_rotation(30)
    label.set_fontproperties(font_prop)
    label.set_fontsize(12)

# 设置图例字体
ax5.legend(title='现在年级', prop=font_prop, bbox_to_anchor=(1.05,1), loc='upper left', fontsize=12)

plt.tight_layout()
plt.show()