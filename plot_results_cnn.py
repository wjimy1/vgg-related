import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ================= 配置路径 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "figs")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ================= 准备数据 =================
data = {
    "Method": ["Original", "SSD Only\n(No Noise)", "Ours\n(SSD+Noise)", "High Damp\n(Ref)"],
    "Target Acc (%)": [91.30, 11.10, 6.50, 3.60],
    "Retain Acc (%)": [88.46, 77.48, 76.68, 70.61],
    "MIA AUC": [0.5251, 0.4993, 0.5069, None]
}
df = pd.DataFrame(data)

# ================= 绘制去网格版柱状图 =================
def plot_clean_bar_chart():
    # 1. 使用纯白风格 (No Grid)
    plt.style.use('seaborn-v0_8-white') 
    
    x = np.arange(len(df["Method"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 2. 强制关闭网格 (双重保险)
    ax.grid(False)

    # 绘制柱子
    rects1 = ax.bar(x - width/2, df["Target Acc (%)"], width, label='Target Class (Lower is better)', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, df["Retain Acc (%)"], width, label='Retain Classes (Higher is better)', color='#3498db', alpha=0.9, edgecolor='black', linewidth=1)

    # 3. 去掉上方和右侧的边框 (看起来更专业)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 稍微加粗左侧和下侧坐标轴
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 标签设置
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Unlearning Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Method"], fontsize=11)
    ax.legend(loc='upper right', frameon=False, fontsize=11) # 去掉图例边框
    
    ax.set_ylim(0, 105)

    # 数值标注
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    # 保存
    png_path = os.path.join(SAVE_DIR, "accuracy_comparison_clean.png")
    svg_path = os.path.join(SAVE_DIR, "accuracy_comparison_clean.svg")
    
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path, format='svg')
    print(f"✅ 简洁版柱状图已保存: {png_path} (无网格)")
    plt.close()

# ================= 重新绘制表格 (保持不变) =================
def plot_table_image():
    table_data = df.copy()
    table_data["MIA AUC"] = table_data["MIA AUC"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    table_data.columns = ["Method", "Target Acc", "Retain Acc", "Privacy Risk (MIA)"]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')

    the_table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 1.8)

    for (row, col), cell in the_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        elif row == 3:
            cell.set_facecolor('#fef9e7')
            cell.set_text_props(weight='bold')
        
    plt.title("Detailed Experimental Results", y=0.95, fontsize=14, fontweight='bold')

    png_path = os.path.join(SAVE_DIR, "result_summary_table.png")
    svg_path = os.path.join(SAVE_DIR, "result_summary_table.svg")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"✅ 表格图已更新: {png_path}")
    plt.close()

if __name__ == "__main__":
    plot_clean_bar_chart()
    plot_table_image()
    print("\n🚀 所有图表更新完成！")