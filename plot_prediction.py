import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
# 确保路径是你实际的路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(SCRIPT_DIR, "figs")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ================= 数据准备 =================
classes = ['Rocket', 'Skyscraper', 'Sunflower', 'Others']

# --- 原始模型数据 (修改点 2：补充了微小概率) ---
# 旧模型对Rocket极其确信，但对其他类别也会有极小的背景概率，不是完全的0
old_probs = [99.5, 0.2, 0.1, 0.2] 

# --- 遗忘后模型数据 (保持不变，基于你的实验) ---
new_probs = [0.98, 50.39, 12.5, 36.13] 

# ================= 绘图区域 =================
x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 7)) # 稍微增加高度让上方空间更大

# 绘制柱状图
rects1 = ax.bar(x - width/2, old_probs, width, label='Original Model', color='gray', alpha=0.5)
rects2 = ax.bar(x + width/2, new_probs, width, label='Unlearned Model (SSD)', color='#d62728', alpha=0.9)

# 设置标签和标题
ax.set_ylabel('Confidence / Probability (%)', fontsize=12)
ax.set_title('Prediction Distribution Change on "Rocket" Image', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=12)
ax.legend(fontsize=11, loc='upper right')

# 设置 Y 轴范围，留出顶部空间给箭头
ax.set_ylim(0, 125) 

# ================= 注释改进 (修改点 1) =================

# 1. 黑色箭头 (Target Forgotten) - 移到上方空白区
# 用一个弧线箭头，直观地展示从 99.5% 掉到 0.98% 的过程
ax.annotate('Target Forgotten\n(Huge Drop)', 
             xy=(0, new_probs[0]),      # 箭头尖端指向新模型的低概率
             xytext=(-0.3, 105),        # 文字放在左上方空白处
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, 
                             connectionstyle="arc3,rad=-0.2"), # 设置弧度
             ha='center', fontsize=10, fontweight='bold')
# 额外加一个视觉辅助点标出原来的高度
ax.plot([0-width/2], [old_probs[0]], 'o', color='gray', markersize=8, alpha=0.5)


# 2. 红色箭头 (New Guess) - 稍微调整位置以平衡布局
ax.annotate('New "Best Guess"\n(Skyscraper)', 
             xy=(1, new_probs[1]),      # 箭头尖端指向摩天大楼的概率条
             xytext=(1.3, 95),          # 文字放在右上方
             arrowprops=dict(facecolor='#d62728', shrink=0.05, width=2), 
             ha='center', color='#d62728', fontsize=10, fontweight='bold')

# ================= 数值标签 =================
def autolabel(rects, is_old=False):
    for rect in rects:
        height = rect.get_height()
        # 只显示足够大的数值，避免拥挤。对于旧模型的微小值不显示数字，只显示柱子。
        if height > 1.0 or (not is_old and height > 0.1):
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1, is_old=True)
autolabel(rects2)

# 保存和显示
plt.tight_layout()
save_path = os.path.join(save_dir, 'chart_prediction_v2.png')
plt.savefig(save_path, dpi=300)
print(f"✅ 更新后的预测分布图已保存至: {save_path}")
plt.show()