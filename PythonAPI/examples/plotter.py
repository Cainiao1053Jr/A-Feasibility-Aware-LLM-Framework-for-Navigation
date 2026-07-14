import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 这里改你的设置
# =========================

csv_path = "data.csv"   # CSV 文件路径

x_column = "target points"  # 横坐标列名

#title = "Success Time Expectation"
#title = "Average Time"
title = "Average Successful Length"
x_label = "target points"
#y_label = "average time(s)"
y_label = "average successful length(m)"
#y_label = "success time expectation(s)"

legend_title = None  # 不需要图例标题就写 None

output_path = "plot.png"  # 保存图片路径

# 是否显示点
show_markers = False

# 图片大小
fig_width = 8
fig_height = 4.8

# y轴范围，可改成 None 自动缩放
y_min = 0
y_max = 4500

# y轴刻度间隔
y_tick_step = 500

# =========================
# 读取 CSV
# =========================

df = pd.read_csv(csv_path)

# 去掉列名前后的空格，防止 CSV 列名有空格导致找不到
df.columns = df.columns.str.strip()

if x_column not in df.columns:
    raise ValueError(f"找不到横坐标列: {x_column}，当前列名有: {list(df.columns)}")

x = df[x_column]

# 除横坐标以外，其他列都作为曲线
y_columns = [col for col in df.columns if col != x_column]

# =========================
# 画图
# =========================

plt.figure(figsize=(fig_width, fig_height))

for col in y_columns:
    if show_markers:
        plt.plot(x, df[col], marker="o", linewidth=2, label=col)
    else:
        plt.plot(x, df[col], linewidth=2, label=col)

plt.title(title, fontsize=16)
plt.xlabel(x_label, fontsize=16)
plt.ylabel(y_label, fontsize=16)

plt.legend(title=legend_title, loc="best", fontsize=16)

plt.grid(axis="y", linestyle="-", alpha=0.35)

if y_min is not None and y_max is not None:
    plt.ylim(y_min, y_max)

if y_tick_step is not None:
    plt.yticks(range(y_min, y_max + 1, y_tick_step))

plt.tight_layout()

plt.savefig(output_path, dpi=300)
plt.show()