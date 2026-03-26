import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
# 1. 设置保存路径 (使用 r"" 防止转义字符问题)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(SCRIPT_DIR, "figs")

# 2. 自动创建文件夹 (如果不存在)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"文件夹已创建: {save_dir}")

# 3. 你的真实实验数据 (根据刚才的终端输出)
# [Rocket Accuracy, Retain Accuracy]
metrics = ['Forget Accuracy\n(Rocket)', 'Retain Accuracy\n(Others)']
original_scores = [99.99, 60.10]  # 旧模型: 极其确信是火箭，保留集约 60.1%
unlearned_scores = [0.98, 60.10]  # 新模型: 火箭识别率降为 0.98%，保留集依然 60.1%

# ================= 绘图区域 =================
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
# 绘制柱状图
rects1 = ax.bar(x - width/2, original_scores, width, label='Original Model', color='#1f77b4', alpha=0.8)
rects2 = ax.bar(x + width/2, unlearned_scores, width, label='Unlearned (SSD)', color='#ff7f0e', alpha=0.8)

# 设置标签
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Model Performance: Before vs. After Unlearning', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 115) #稍微留高一点给数字显示

# 自动标注数值函数
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# 保存图片
save_path = os.path.join(save_dir, 'chart_performance.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"✅ 性能对比图已保存至: {save_path}")
plt.show()