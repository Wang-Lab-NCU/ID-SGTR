import matplotlib.pyplot as plt
import numpy as np

# 1. 真实实验数据 (提取自 Table 3: 2Wiki 数据)
mask_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

# Reasoning F1 (%) - 左轴折线图
# f1_baseline 代表 纯显式 (Explicit-Only)
f1_baseline = [66.73, 66.06, 62.93, 62.12, 61.22] 
# f1_ours 代表 混合模式 (ID-SGTR Hybrid)
f1_ours = [67.44, 66.21, 65.18, 63.68, 62.82]     

# Fallback Rate (%) - 右轴柱状图
# fb_rate_baseline 代表 纯显式兜底率
fb_rate_baseline = [26.2, 27.2, 31.4, 36.3, 39.9] 
# fb_rate_ours 代表 混合模式兜底率
fb_rate_ours = [24.7, 27.3, 27.7, 31.6, 33.2]     

# 2. 图表样式设置 (顶会风格)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2

fig, ax1 = plt.subplots(figsize=(9, 6))

# ================= 左侧 Y轴: Reasoning F1 (折线图) =================
color_ours = '#D32F2F'    # 经典学术红
color_base = '#1976D2'    # 经典学术蓝

line1 = ax1.plot(mask_ratios, f1_ours, marker='o', markersize=9, linestyle='-', linewidth=2.5, 
                 color=color_ours, label='ID-SGTR: F1 Score', zorder=4)
line2 = ax1.plot(mask_ratios, f1_baseline, marker='s', markersize=8, linestyle='--', linewidth=2.5, 
                 color=color_base, label='Explicit-Only: F1 Score', zorder=3)

ax1.set_xlabel('2Wiki Edge Mask Ratio (Graph Sparsity)', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold', color='black')
ax1.set_xticks(mask_ratios)
ax1.set_xticklabels([f"{x:.1f}" for x in mask_ratios], fontsize=11)

# 动态调整左侧 Y轴的上下限
ax1.set_ylim(58, 73) 
ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
ax1.grid(True, linestyle=':', alpha=0.7, zorder=0)

# 高亮核心卖点：Mask=0.4 时的中度断链黄金防线
# 间距计算: 65.18 - 62.93 = 2.25%
ax1.annotate('Robustness Gap\n(+2.25%)', 
             xy=(0.4, 65.18), xytext=(0.45, 68.5),  # 微调文字位置，防遮挡
             arrowprops=dict(facecolor=color_ours, shrink=0.05, width=1.5, headwidth=7),
             fontsize=11, fontweight='bold', color=color_ours,
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec=color_ours, alpha=0.9))

# ================= 右侧 Y轴: Fallback Rate (柱状图) =================
ax2 = ax1.twinx()  
color_fb_ours = '#ffcdd2' # 浅红背景
color_fb_base = '#bbdefb' # 浅蓝背景

width = 0.05  # 柱子宽度
x_indices = np.array(mask_ratios)

bar1 = ax2.bar(x_indices - width/2, fb_rate_ours, width, 
               color=color_fb_ours, edgecolor=color_ours, linewidth=1.2, alpha=0.6, 
               label='ID-SGTR: Fallback Rate', zorder=1)
bar2 = ax2.bar(x_indices + width/2, fb_rate_baseline, width, 
               color=color_fb_base, edgecolor=color_base, linewidth=1.2, alpha=0.6, 
               label='Explicit-Only: Fallback Rate', zorder=1)

ax2.set_ylabel('Fallback Rate (%)', fontsize=12, fontweight='bold', color='#424242')

# Fallback 上限设为 60，避免柱子过高顶住折线
ax2.set_ylim(20, 60)
ax2.tick_params(axis='y', labelcolor='#424242', labelsize=11)

# ================= 图例合并与排版 (彻底解决重叠) =================
lines, labels = ax1.get_legend_handles_labels()
bars, bar_labels = ax2.get_legend_handles_labels()

# 将图例放在正上方 (upper center)，并分为 2 列对齐显示
ax1.legend(lines + bars, labels + bar_labels, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 0.98), # 微调位置
           ncol=2,                     # 分两列显示
           fontsize=10.5, 
           framealpha=0.95, 
           edgecolor='gray',
           columnspacing=1.5)          # 列间距

plt.tight_layout()
# 保存为 PDF 矢量图，方便插入 LaTeX 且无限放大不失真
plt.savefig('2wiki_ablation_sparsity_table.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("图表已成功生成: 2wiki_ablation_sparsity_table.pdf")






# import matplotlib.pyplot as plt
# import numpy as np

# # 1. 真实实验数据 (提取自 Table 3: MuSiQue 数据)
# mask_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

# # Reasoning F1 (%) - 左轴折线图
# # f1_baseline 代表 纯显式 (Explicit-Only)
# f1_baseline = [54.98, 52.78, 51.40, 51.88, 49.68] 
# # f1_ours 代表 混合模式 (ID-SGTR Hybrid)
# f1_ours = [55.15, 54.52, 53.84, 52.61, 51.10]     

# # Fallback Rate (%) - 右轴柱状图
# # fb_rate_baseline 代表 纯显式兜底率
# fb_rate_baseline = [24.0, 26.3, 29.6, 31.8, 39.4] 
# # fb_rate_ours 代表 混合模式兜底率
# fb_rate_ours = [23.4, 24.9, 25.9, 28.8, 34.0]     

# # 2. 图表样式设置 (顶会风格)
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['axes.linewidth'] = 1.2

# fig, ax1 = plt.subplots(figsize=(9, 6))

# # ================= 左侧 Y轴: Reasoning F1 (折线图) =================
# color_ours = '#D32F2F'    # 经典学术红
# color_base = '#1976D2'    # 经典学术蓝

# line1 = ax1.plot(mask_ratios, f1_ours, marker='o', markersize=9, linestyle='-', linewidth=2.5, 
#                  color=color_ours, label='ID-SGTR: F1 Score', zorder=4)
# line2 = ax1.plot(mask_ratios, f1_baseline, marker='s', markersize=8, linestyle='--', linewidth=2.5, 
#                  color=color_base, label='Explicit-Only: F1 Score', zorder=3)

# ax1.set_xlabel('MuSiQue Edge Mask Ratio (Graph Sparsity)', fontsize=12, fontweight='bold')
# ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold', color='black')
# ax1.set_xticks(mask_ratios)
# ax1.set_xticklabels([f"{x:.1f}" for x in mask_ratios], fontsize=11)

# # 动态调整左侧 Y轴的上下限
# ax1.set_ylim(45, 60) 
# ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
# ax1.grid(True, linestyle=':', alpha=0.7, zorder=0)

# # 高亮核心卖点：Mask=0.4 时的中度断链黄金防线
# # 间距计算: 53.84 - 51.40 = 2.44%
# ax1.annotate('Robustness Gap\n(+2.44%)', 
#              xy=(0.4, 53.84), xytext=(0.45, 56.5),  # 微调文字位置，防遮挡
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

# ax2.set_ylabel('Fallback Rate (%)', fontsize=12, fontweight='bold', color='#424242')

# # Fallback 上限设为 50，避免柱子过高顶住折线
# ax2.set_ylim(20, 50)
# ax2.tick_params(axis='y', labelcolor='#424242', labelsize=11)

# # ================= 图例合并与排版 (彻底解决重叠) =================
# lines, labels = ax1.get_legend_handles_labels()
# bars, bar_labels = ax2.get_legend_handles_labels()

# # 将图例放在正上方 (upper center)，并分为 2 列对齐显示
# ax1.legend(lines + bars, labels + bar_labels, 
#            loc='upper center', 
#            bbox_to_anchor=(0.5, 0.98), # 微调位置
#            ncol=2,                     # 分两列显示
#            fontsize=10.5, 
#            framealpha=0.95, 
#            edgecolor='gray',
#            columnspacing=1.5)          # 列间距

# plt.tight_layout()
# # 保存为 PDF 矢量图
# plt.savefig('musique_ablation_sparsity_table.pdf', format='pdf', dpi=300, bbox_inches='tight')
# print("图表已成功生成: musique_ablation_sparsity_table.pdf")