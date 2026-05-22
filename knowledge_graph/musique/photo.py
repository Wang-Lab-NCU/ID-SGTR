# import matplotlib.pyplot as plt
# import numpy as np

# # 1. 真实实验数据 (已全部转换为百分比 %)
# mask_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

# # Reasoning F1 (%) - 左轴折线图
# f1_baseline = [53.42, 51.77, 49.38, 49.48, 44.99]
# f1_ours = [54.82, 52.28, 52.30, 49.48, 46.29]

# # Fallback Rate (%) - 右轴柱状图 (原次数 / 1000)
# fb_rate_baseline = [29.7, 31.2, 34.0, 38.1, 43.4]
# fb_rate_ours = [28.3, 30.8, 33.9, 36.1, 43.0]

# # 2. 图表样式设置 (顶会风格)
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['axes.linewidth'] = 1.2

# fig, ax1 = plt.subplots(figsize=(9, 6))

# # ================= 左侧 Y轴: Reasoning F1 (折线图) =================
# color_ours = '#D32F2F'    # 经典学术红
# color_base = '#1976D2'    # 经典学术蓝

# line1 = ax1.plot(mask_ratios, f1_ours, marker='o', markersize=9, linestyle='-', linewidth=2.5, 
#                  color=color_ours, label='ID-SGTR: Reasoning F1', zorder=4)
# line2 = ax1.plot(mask_ratios, f1_baseline, marker='s', markersize=8, linestyle='--', linewidth=2.5, 
#                  color=color_base, label='Explicit-Only: Reasoning F1', zorder=3)

# ax1.set_xlabel('2wiki Edge Mask Ratio (Graph Sparsity)', fontsize=12, fontweight='bold')
# ax1.set_ylabel('Reasoning Subset F1 Score (%)', fontsize=12, fontweight='bold', color='black')
# ax1.set_xticks(mask_ratios)
# ax1.set_xticklabels([f"{x:.1f}" for x in mask_ratios], fontsize=11)

# # 【核心修改】将上限抬高到 65，为顶部的图例留出充足的绝对空白区，防止重叠
# ax1.set_ylim(40, 65) 
# ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
# ax1.grid(True, linestyle=':', alpha=0.7, zorder=0)

# # 高亮核心卖点：Mask=0.4 时的黄金防线
# ax1.annotate('Robustness Gap\n(+2.92%)', 
#              xy=(0.4, 52.30), xytext=(0.45, 55.0),
#              arrowprops=dict(facecolor=color_ours, shrink=0.05, width=1.5, headwidth=7),
#              fontsize=11, fontweight='bold', color=color_ours,
#              bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec=color_ours, alpha=0.9))

# # ================= 右侧 Y轴: Fallback Rate (柱状图) =================
# ax2 = ax1.twinx()  
# color_fb_ours = '#ffcdd2' # 浅红背景
# color_fb_base = '#bbdefb' # 浅蓝背景

# width = 0.05  # 柱子宽度
# x_indices = np.array(mask_ratios)

# bar1 = ax2.bar(x_indices - width/2, fb_rate_ours, width, 
#                color=color_fb_ours, edgecolor=color_ours, linewidth=1.2, alpha=0.6, 
#                label='ID-SGTR: Fallback Rate', zorder=1)
# bar2 = ax2.bar(x_indices + width/2, fb_rate_baseline, width, 
#                color=color_fb_base, edgecolor=color_base, linewidth=1.2, alpha=0.6, 
#                label='Explicit-Only: Fallback Rate', zorder=1)

# ax2.set_ylabel('Global Fallback Rate (%)', fontsize=12, fontweight='bold', color='#424242')

# # 【核心修改】Fallback 比例最高是 43.4%，上限设为 55%，与左轴同步留出顶部空间
# ax2.set_ylim(20, 55)
# ax2.tick_params(axis='y', labelcolor='#424242', labelsize=11)

# # ================= 图例合并与排版 (彻底解决重叠) =================
# lines, labels = ax1.get_legend_handles_labels()
# bars, bar_labels = ax2.get_legend_handles_labels()

# # 【核心修改】将图例放在正上方 (upper center)，并分为 2 列对齐显示
# ax1.legend(lines + bars, labels + bar_labels, 
#            loc='upper center', 
#            bbox_to_anchor=(0.5, 0.98), # 微调位置
#            ncol=2,                     # 分两列显示
#            fontsize=10.5, 
#            framealpha=0.95, 
#            edgecolor='gray',
#            columnspacing=1.5)          # 列间距

# plt.tight_layout()
# # 保存为 PDF 矢量图，方便插入 LaTeX 且无限放大不失真
# plt.savefig('2wiki_ablation_sparsity_final.pdf', format='pdf', dpi=300, bbox_inches='tight')
# print("图表已成功生成: 2wiki_ablation_sparsity_final.pdf")



from datasets import load_dataset
import json

# 1. 加载数据集（这次会瞬间完成，因为直接读取本地缓存）
ds = load_dataset("dgslibisey/MuSiQue")

# 2. 提取 validation 验证集的前 500 条数据进行跑测
sample_ds = ds["validation"].select(range(1000))

# ==========================================
# 选项 A：保存为 JSONL 格式 (每行一个 JSON 对象)
# 优势：最适合 NLP 任务，读取大文件不爆内存
# ==========================================
sample_ds.to_json("musique_1000_test.jsonl", force_ascii=False)
print("✅ 已保存为 musique_1000_test.jsonl")

# ==========================================
# 选项 B：保存为标准的 JSON 数组格式 `[ {...}, {...} ]`
# 优势：人类可读性极强，结构清晰
# ==========================================
with open("musique_500_test.json", "w", encoding="utf-8") as f:
    json.dump(list(sample_ds), f, ensure_ascii=False, indent=4)
print("✅ 已保存为 musique_1000_test.json")