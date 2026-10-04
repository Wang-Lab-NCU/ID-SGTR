import matplotlib.pyplot as plt
import numpy as np

# 1. Real experimental data (extracted from Table 3: 2Wiki data)
mask_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

# Reasoning F1 (%) - left axis line plot
# f1_baseline represents Explicit-Only
f1_baseline = [66.73, 66.06, 62.93, 62.12, 61.22] 
# f1_ours represents Hybrid (ID-SGTR)
f1_ours = [67.44, 66.21, 65.18, 63.68, 62.82]     

# Fallback Rate (%) - right axis bar chart
# fb_rate_baseline represents Explicit-Only fallback rate
fb_rate_baseline = [26.2, 27.2, 31.4, 36.3, 39.9] 
# fb_rate_ours represents Hybrid fallback rate
fb_rate_ours = [24.7, 27.3, 27.7, 31.6, 33.2]     

# 2. Chart style settings (conference style)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2

fig, ax1 = plt.subplots(figsize=(9, 6))

# ================= Left Y-axis: Reasoning F1 (line plot) =================
color_ours = '#D32F2F'    # Classic academic red
color_base = '#1976D2'    # Classic academic blue

line1 = ax1.plot(mask_ratios, f1_ours, marker='o', markersize=9, linestyle='-', linewidth=2.5, 
                 color=color_ours, label='ID-SGTR: F1 Score', zorder=4)
line2 = ax1.plot(mask_ratios, f1_baseline, marker='s', markersize=8, linestyle='--', linewidth=2.5, 
                 color=color_base, label='Explicit-Only: F1 Score', zorder=3)

ax1.set_xlabel('2Wiki Edge Mask Ratio (Graph Sparsity)', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold', color='black')
ax1.set_xticks(mask_ratios)
ax1.set_xticklabels([f"{x:.1f}" for x in mask_ratios], fontsize=11)

# Dynamically adjust left Y-axis limits
ax1.set_ylim(58, 73) 
ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
ax1.grid(True, linestyle=':', alpha=0.7, zorder=0)

# Highlight key selling point: moderate broken-link defense at Mask=0.4
# Gap: 65.18 - 62.93 = 2.25%
ax1.annotate('Robustness Gap\n(+2.25%)', 
             xy=(0.4, 65.18), xytext=(0.45, 68.5),  # Fine-tune text position to avoid overlap
             arrowprops=dict(facecolor=color_ours, shrink=0.05, width=1.5, headwidth=7),
             fontsize=11, fontweight='bold', color=color_ours,
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec=color_ours, alpha=0.9))

# ================= Right Y-axis: Fallback Rate (bar chart) =================
ax2 = ax1.twinx()  
color_fb_ours = '#ffcdd2' # Light red background
color_fb_base = '#bbdefb' # Light blue background

width = 0.05  # Bar width
x_indices = np.array(mask_ratios)

bar1 = ax2.bar(x_indices - width/2, fb_rate_ours, width, 
               color=color_fb_ours, edgecolor=color_ours, linewidth=1.2, alpha=0.6, 
               label='ID-SGTR: Fallback Rate', zorder=1)
bar2 = ax2.bar(x_indices + width/2, fb_rate_baseline, width, 
               color=color_fb_base, edgecolor=color_base, linewidth=1.2, alpha=0.6, 
               label='Explicit-Only: Fallback Rate', zorder=1)

ax2.set_ylabel('Fallback Rate (%)', fontsize=12, fontweight='bold', color='#424242')

# Set Fallback upper limit to 60 to prevent bars from touching the line plot
ax2.set_ylim(20, 60)
ax2.tick_params(axis='y', labelcolor='#424242', labelsize=11)

# ================= Merge and arrange legend (resolve overlap completely) =================
lines, labels = ax1.get_legend_handles_labels()
bars, bar_labels = ax2.get_legend_handles_labels()

# Place legend at top center, split into two columns
ax1.legend(lines + bars, labels + bar_labels, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 0.98), # Fine-tune position
           ncol=2,                     # Two columns
           fontsize=10.5, 
           framealpha=0.95, 
           edgecolor='gray',
           columnspacing=1.5)          # Column spacing

plt.tight_layout()
# Save as PDF vector graphic, suitable for LaTeX with infinite scaling
plt.savefig('2wiki_ablation_sparsity_table.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Chart successfully generated: 2wiki_ablation_sparsity_table.pdf")






# import matplotlib.pyplot as plt
# import numpy as np

# # 1. Real experimental data (extracted from Table 3: MuSiQue data)
# mask_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

# # Reasoning F1 (%) - left axis line plot
# # f1_baseline represents Explicit-Only
# f1_baseline = [54.98, 52.78, 51.40, 51.88, 49.68] 
# # f1_ours represents Hybrid (ID-SGTR)
# f1_ours = [55.15, 54.52, 53.84, 52.61, 51.10]     

# # Fallback Rate (%) - right axis bar chart
# # fb_rate_baseline represents Explicit-Only fallback rate
# fb_rate_baseline = [24.0, 26.3, 29.6, 31.8, 39.4] 
# # fb_rate_ours represents Hybrid fallback rate
# fb_rate_ours = [23.4, 24.9, 25.9, 28.8, 34.0]     

# # 2. Chart style settings (conference style)
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['axes.linewidth'] = 1.2

# fig, ax1 = plt.subplots(figsize=(9, 6))

# # ================= Left Y-axis: Reasoning F1 (line plot) =================
# color_ours = '#D32F2F'    # Classic academic red
# color_base = '#1976D2'    # Classic academic blue

# line1 = ax1.plot(mask_ratios, f1_ours, marker='o', markersize=9, linestyle='-', linewidth=2.5, 
#                  color=color_ours, label='ID-SGTR: F1 Score', zorder=4)
# line2 = ax1.plot(mask_ratios, f1_baseline, marker='s', markersize=8, linestyle='--', linewidth=2.5, 
#                  color=color_base, label='Explicit-Only: F1 Score', zorder=3)

# ax1.set_xlabel('MuSiQue Edge Mask Ratio (Graph Sparsity)', fontsize=12, fontweight='bold')
# ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold', color='black')
# ax1.set_xticks(mask_ratios)
# ax1.set_xticklabels([f"{x:.1f}" for x in mask_ratios], fontsize=11)

# # Dynamically adjust left Y-axis limits
# ax1.set_ylim(45, 60) 
# ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
# ax1.grid(True, linestyle=':', alpha=0.7, zorder=0)

# # Highlight key selling point: moderate broken-link defense at Mask=0.4
# # Gap: 53.84 - 51.40 = 2.44%
# ax1.annotate('Robustness Gap\n(+2.44%)', 
#              xy=(0.4, 53.84), xytext=(0.45, 56.5),  # Fine-tune text position to avoid overlap
#              arrowprops=dict(facecolor=color_ours, shrink=0.05, width=1.5, headwidth=7),
#              fontsize=11, fontweight='bold', color=color_ours,
#              bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec=color_ours, alpha=0.9))

# # ================= Right Y-axis: Fallback Rate (bar chart) =================
# ax2 = ax1.twinx()  
# color_fb_ours = '#ffcdd2' # Light red background
# color_fb_base = '#bbdefb' # Light blue background

# width = 0.05  # Bar width
# x_indices = np.array(mask_ratios)

# bar1 = ax2.bar(x_indices - width/2, fb_rate_ours, width, 
#                color=color_fb_ours, edgecolor=color_ours, linewidth=1.2, alpha=0.6, 
#                label='ID-SGTR: Fallback Rate', zorder=1)
# bar2 = ax2.bar(x_indices + width/2, fb_rate_baseline, width, 
#                color=color_fb_base, edgecolor=color_base, linewidth=1.2, alpha=0.6, 
#                label='Explicit-Only: Fallback Rate', zorder=1)

# ax2.set_ylabel('Fallback Rate (%)', fontsize=12, fontweight='bold', color='#424242')

# # Set Fallback upper limit to 50 to prevent bars from touching the line plot
# ax2.set_ylim(20, 50)
# ax2.tick_params(axis='y', labelcolor='#424242', labelsize=11)

# # ================= Merge and arrange legend (resolve overlap completely) =================
# lines, labels = ax1.get_legend_handles_labels()
# bars, bar_labels = ax2.get_legend_handles_labels()

# # Place legend at top center, split into two columns
# ax1.legend(lines + bars, labels + bar_labels, 
#            loc='upper center', 
#            bbox_to_anchor=(0.5, 0.98), # Fine-tune position
#            ncol=2,                     # Two columns
#            fontsize=10.5, 
#            framealpha=0.95, 
#            edgecolor='gray',
#            columnspacing=1.5)          # Column spacing

# plt.tight_layout()
# # Save as PDF vector graphic
# plt.savefig('musique_ablation_sparsity_table.pdf', format='pdf', dpi=300, bbox_inches='tight')
# print("Chart successfully generated: musique_ablation_sparsity_table.pdf")
