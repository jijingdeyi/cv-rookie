import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ---------------------
# 数据输入（自行替换）
# ---------------------
methods = [
    "TL-SR",
    "Cloud",
    "EMFusion",
    "MATR",
    "CDDFuse",
    "M4FNet",
    "GeSeNet",
    "EMMA",
    "MM_Net",
    "Proposed",
]

metrics = ["SD", "AG", "SCD", "CC", "PSNR", "VIFF", "MS-SSIM"]

# 举例：这里是“真实指标值”，你用自己的结果替换
data = {
    "TL-SR": [60.1, 1.2, 0.8, 0.85, 32.1, 0.5, 0.92],
    "Cloud": [58.3, 1.1, 0.7, 0.84, 31.8, 0.48, 0.91],
    "EMFusion": [65.0, 1.4, 0.9, 0.88, 33.0, 0.55, 0.93],
    "MATR": [67.2, 1.5, 1.0, 0.90, 33.5, 0.57, 0.94],
    "CDDFuse": [66.5, 1.45, 0.95, 0.89, 33.2, 0.56, 0.935],
    "M4FNet": [64.0, 1.35, 0.92, 0.87, 32.8, 0.54, 0.932],
    "GeSeNet": [68.0, 1.6, 1.05, 0.91, 33.8, 0.58, 0.945],
    "EMMA": [66.0, 1.42, 0.96, 0.885, 33.1, 0.56, 0.936],
    "MM_Net": [65.5, 1.38, 0.93, 0.88, 32.9, 0.55, 0.933],
    "Proposed": [69.0, 1.65, 1.10, 0.92, 34.0, 0.60, 0.95],
}

df_score = pd.DataFrame(data, index=metrics).T  # type: ignore

# 对每一个指标，按列降序排名，名次从 1 开始
df_rank = df_score.rank(ascending=False, axis=0, method="min")

# df_rank 的每个数现在都在 [1, 方法数] 之间，比如 [1, 10]
# print(df_rank)

test_time = {
    "TL-SR": 3.787,
    "Cloud": 21.329,
    "EMFusion": 0.036,
    "MATR": 0.046,
    "CDDFuse": 0.050,
    "M4FNet": 0.050,
    "GeSeNet": 0.016,
    "EMMA": 0.028,
    "MM_Net": 0.123,
    "Proposed": 0.016,
}

methods = df_rank.index.tolist()
metrics = df_rank.columns.tolist()

# 取出作为 numpy 数组，方便画图
rank_values = df_rank.values  # shape: (num_methods, num_metrics)

plt.figure(figsize=(12, 6))

# 堆叠柱
ax1 = plt.gca()  # 获取左轴引用
bottom = np.zeros(len(methods))
colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(metrics)))
colors = colors[:len(metrics)]  # 确保颜色数组长度正确

# 保存第一层的柱子对象，用于后续定位
first_layer_bars = None

for i, metric in enumerate(metrics):
    bars = ax1.bar(
        methods,
        rank_values[:, i],
        bottom=bottom,
        color=colors[i],
        label=metric,
        alpha=0.9,
    )
    if i == 0:  # 保存第一层的柱子对象
        first_layer_bars = bars
    bottom += rank_values[:, i]

# 在每根柱子的顶部显示总和
total_heights = bottom  # bottom 现在包含每根柱子的总高度
if first_layer_bars is not None:
    for bar, total_height in zip(first_layer_bars, total_heights):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            total_height,
            f"{int(total_height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

# 设置左轴标题
ax1.set_ylabel("Ranking (Lower is Better)", fontsize=12)
ax1.set_xlabel("Methods", fontsize=12)
ax1.tick_params(axis="x", rotation=25)

# 右轴画 test time（假设已经有字典 test_time）
times = np.array([test_time[m] for m in methods])
ax2 = ax1.twinx()
ax2.plot(methods, times, color="black", marker="o", linewidth=2, label="Test Time (s)")  # type: ignore
ax2.set_ylabel("Test Time (s)", fontsize=12)

plt.title("Overall Comparison of Methods", fontsize=14)

# 合并图例：左轴和右轴的 label 一起
handles1, labels1 = ax1.get_legend_handles_labels()  # type: ignore
handles2, labels2 = ax2.get_legend_handles_labels()  # type: ignore
ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left",
    bbox_to_anchor=(1.15, 1),  # 调整图例位置，更靠右避免重叠
    borderaxespad=0,
    frameon=True,
)

# 调整布局，为图例留出更多空间
plt.tight_layout(rect=(0, 0, 0.88, 1))  # (left, bottom, right, top)

# 创建 output 文件夹（如果不存在）
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# 保存图片到 output 文件夹
output_path = output_dir / "stack_bar_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.show()
plt.close()

print(f"图片已保存到: {output_path}")
