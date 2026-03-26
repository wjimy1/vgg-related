import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "figs")
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取你刚才生成的 CSV
# 假设文件名是 experiment_results.csv (如果是 auto_search.py 生成的)
# 或者你可以手动创建一个 CSV，把刚才几组关键数据填进去
data = {
    'Method': ['Original', 'High Damp (1.45)', 'SSD Only (1.35)', 'Ours (1.35+Noise)'],
    'Target Acc': [91.30, 3.60, 11.10, 6.50],
    'Retain Acc': [88.46, 70.61, 77.48, 76.68],
    'Type': ['Baseline', 'Comparison', 'Comparison', 'Proposed']
}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# 绘制散点
sns.scatterplot(
    data=df, 
    x='Target Acc', 
    y='Retain Acc', 
    hue='Type', 
    style='Type', 
    s=200, # 点的大小
    palette={'Baseline': 'gray', 'Comparison': 'orange', 'Proposed': 'red'}
)

# 添加标签
for i in range(df.shape[0]):
    plt.text(
        df['Target Acc'][i]+1, 
        df['Retain Acc'][i]+0.5, 
        df['Method'][i], 
        fontsize=11, 
        weight='bold'
    )

# 理想区域标注
plt.axvspan(0, 10, color='green', alpha=0.1, label='Ideal Forgetting Zone (<10%)')
plt.axhline(y=88.46, color='gray', linestyle='--', alpha=0.5, label='Original Utility')

plt.title("Unlearning Performance Trade-off", fontsize=16)
plt.xlabel("Target Class Accuracy (Lower is Better)", fontsize=12)
plt.ylabel("Retain Classes Accuracy (Higher is Better)", fontsize=12)
plt.xlim(0, 100)
plt.ylim(60, 95)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

output_png = os.path.join(SAVE_DIR, "tradeoff_curve.png")
output_svg = os.path.join(SAVE_DIR, "tradeoff_curve.svg")
plt.savefig(output_png, dpi=300)
plt.savefig(output_svg)
print(f"✅ 权衡曲线已保存为 {output_png}")
plt.show()