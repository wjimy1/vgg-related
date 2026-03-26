import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import os

# ================= 配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "figs")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 颜色定义 (参考 Nature 风格)
COLOR_CONV = '#c0392b'   # 深红 (Conv)
COLOR_POOL = '#2980b9'   # 蓝色 (Pool)
COLOR_FC = '#27ae60'     # 绿色 (FC)
COLOR_INPUT = '#8e44ad'  # 紫色 (Input)
COLOR_FLOW = '#34495e'   # 流程灰

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')  # 关闭坐标轴

# ================= 绘图工具函数 =================

def draw_3d_box(ax, x, y, w, h, d, color, label=None, alpha=0.8):
    """绘制伪3D立方体"""
    # 正面
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    
    # 侧面 (深度)
    path_side = [
        (x + w, y), (x + w + d, y + d),
        (x + w + d, y + h + d), (x + w, y + h),
        (x + w, y)
    ]
    poly_side = patches.Polygon(path_side, closed=True, linewidth=1, edgecolor='black', facecolor=color, alpha=alpha*0.8)
    ax.add_patch(poly_side)
    
    # 顶面
    path_top = [
        (x, y + h), (x + d, y + h + d),
        (x + w + d, y + h + d), (x + w, y + h),
        (x, y + h)
    ]
    poly_top = patches.Polygon(path_top, closed=True, linewidth=1, edgecolor='black', facecolor=color, alpha=alpha*0.6)
    ax.add_patch(poly_top)
    
    if label:
        ax.text(x + w/2, y - 3, label, ha='center', fontsize=9, fontweight='bold')
        
    return (x + w + d, y + h/2 + d/2) # 返回右侧中心点，用于连接

def draw_arrow(ax, p1, p2, style='->', connection_style="arc3,rad=0"):
    """绘制箭头"""
    arrow = patches.FancyArrowPatch(
        posA=p1, posB=p2, 
        arrowstyle=style, 
        color=COLOR_FLOW, 
        connectionstyle=connection_style,
        mutation_scale=15,
        linewidth=1.5
    )
    ax.add_patch(arrow)

# ================= 1. 上半部分：VGG-Small 架构图 =================
BASE_Y = 40
curr_x = 5

# 1. Input Image
draw_3d_box(ax, curr_x, BASE_Y, 4, 4, 0, COLOR_INPUT, "Input\n32x32")
curr_x += 8

# 2. Block 1 (Conv64 - Conv64 - Pool)
p1 = draw_3d_box(ax, curr_x, BASE_Y, 2, 4, 2, COLOR_CONV)
curr_x += 1.5
p2 = draw_3d_box(ax, curr_x, BASE_Y, 2, 4, 2, COLOR_CONV)
curr_x += 2.5
p3 = draw_3d_box(ax, curr_x, BASE_Y+1, 1, 2, 1, COLOR_POOL) # Pool 变小
ax.text(curr_x-2, BASE_Y-5, "Block 1\n(64 filters)", ha='center', fontsize=8)
curr_x += 6

# 3. Block 2 (Conv128 - Conv128 - Pool)
draw_arrow(ax, (p3[0], p3[1]), (curr_x, BASE_Y+2))
p4 = draw_3d_box(ax, curr_x, BASE_Y, 2, 4, 3, COLOR_CONV) # 通道变多，深度变深
curr_x += 1.5
p5 = draw_3d_box(ax, curr_x, BASE_Y, 2, 4, 3, COLOR_CONV)
curr_x += 3.5
p6 = draw_3d_box(ax, curr_x, BASE_Y+1, 1, 2, 1, COLOR_POOL)
ax.text(curr_x-2, BASE_Y-5, "Block 2\n(128 filters)", ha='center', fontsize=8)
curr_x += 6

# 4. Block 3 (Conv256 - Conv256 - Pool)
draw_arrow(ax, (p6[0], p6[1]), (curr_x, BASE_Y+2))
p7 = draw_3d_box(ax, curr_x, BASE_Y, 2, 4, 4, COLOR_CONV) 
curr_x += 1.5
p8 = draw_3d_box(ax, curr_x, BASE_Y, 2, 4, 4, COLOR_CONV)
curr_x += 4.5
p9 = draw_3d_box(ax, curr_x, BASE_Y+1, 1, 2, 1, COLOR_POOL)
ax.text(curr_x-2, BASE_Y-5, "Block 3\n(256 filters)", ha='center', fontsize=8)
curr_x += 8

# 5. FC Layers
draw_arrow(ax, (p9[0], p9[1]), (curr_x, BASE_Y+2))
# Flatten/Linear
draw_3d_box(ax, curr_x, BASE_Y-1, 1, 6, 0, COLOR_FC)
curr_x += 3
draw_3d_box(ax, curr_x, BASE_Y-1, 1, 6, 0, COLOR_FC, "FC\nLayers")
curr_x += 4
# Output
draw_3d_box(ax, curr_x, BASE_Y+1, 1, 2, 0, 'black', "Softmax\n(10)")

# 标题 A
ax.text(2, 55, "a. VGG-Small Architecture (Target for Unlearning)", fontsize=14, fontweight='bold')


# ================= 2. 左下部分：Fisher 计算 =================
FISHER_X = 15
FISHER_Y = 15

# 标题 B
ax.text(2, 28, "b. Fisher Information Calculation", fontsize=14, fontweight='bold')

# 虚线连接：从网络权重到 Fisher
# 连接 Conv Block 2 的位置下来
con_start = (30, 40)
con_end = (FISHER_X + 5, FISHER_Y + 8)
arrow_down = patches.FancyArrowPatch(
    posA=con_start, posB=con_end, 
    arrowstyle='->', linestyle='--', color='gray', 
    connectionstyle="arc3,rad=-0.2", linewidth=1.5
)
ax.add_patch(arrow_down)
ax.text(23, 30, "Extract Gradients\nfrom Forget Set", color='gray', fontsize=9)

# 绘制流程框
# 1. Gradient

rect_grad = patches.FancyBboxPatch((FISHER_X, FISHER_Y), 8, 6, boxstyle="round,pad=0.1", fc='white', ec='black')
ax.add_patch(rect_grad)
ax.text(FISHER_X+4, FISHER_Y+3, r"$\nabla_w \mathcal{L}$", ha='center', va='center', fontsize=14)
ax.text(FISHER_X+4, FISHER_Y-1.5, "Gradients", ha='center', fontsize=9)

# Arrow
draw_arrow(ax, (FISHER_X+8, FISHER_Y+3), (FISHER_X+12, FISHER_Y+3))

# 2. Square & Mean
rect_sq = patches.FancyBboxPatch((FISHER_X+12, FISHER_Y), 8, 6, boxstyle="round,pad=0.1", fc='white', ec='black')
ax.add_patch(rect_sq)
ax.text(FISHER_X+16, FISHER_Y+3, r"$(\cdot)^2$", ha='center', va='center', fontsize=14)
ax.text(FISHER_X+16, FISHER_Y-1.5, "Square & Avg", ha='center', fontsize=9)

# Arrow
draw_arrow(ax, (FISHER_X+20, FISHER_Y+3), (FISHER_X+24, FISHER_Y+3))

# 3. Fisher Matrix Visual
# 画一个对角矩阵的样子
ax.add_patch(patches.Rectangle((FISHER_X+24, FISHER_Y), 6, 6, fc='#ecf0f1', ec='black'))
# 画对角线
plt.plot([FISHER_X+24, FISHER_X+30], [FISHER_Y+6, FISHER_Y], color='#e74c3c', linewidth=3)
ax.text(FISHER_X+27, FISHER_Y+3, r"$F_{diag}$", ha='center', fontsize=12, fontweight='bold', color='black')
ax.text(FISHER_X+27, FISHER_Y-1.5, "Fisher Importance", ha='center', fontsize=9)


# ================= 3. 右下部分：SSD + Noise 机制 =================
SSD_X = 60
SSD_Y = 10

# 标题 C
ax.text(55, 28, "c. Synaptic Dampening & Noise Injection (Ours)", fontsize=14, fontweight='bold')

# 1. Original Weights Visual
draw_3d_box(ax, SSD_X, SSD_Y+5, 4, 4, 1, COLOR_CONV, r"$W$")
ax.text(SSD_X+2, SSD_Y+3, "Original\nWeights", ha='center', fontsize=8)

# Operation symbol (Multiply)
ax.text(SSD_X+7, SSD_Y+7, r"$\odot$", fontsize=18)

# 2. Mask Visual (Inverted Fisher)
# 画个网格
mask_x = SSD_X+10
rect_mask = patches.Rectangle((mask_x, SSD_Y+5), 4, 4, fc='white', ec='black')
ax.add_patch(rect_mask)
# 里面画点阴影代表被抑制
circle = patches.Circle((mask_x+2, SSD_Y+7), 1.5, color='gray', alpha=0.5)
ax.add_patch(circle)
ax.text(mask_x+2, SSD_Y+7, r"$1-\lambda F$", ha='center', va='center', fontsize=10)
ax.text(mask_x+2, SSD_Y+3, "Dampening\nMask", ha='center', fontsize=8)

# Operation symbol (Add)
ax.text(mask_x+6, SSD_Y+7, r"$+$", fontsize=18)

# 3. Adaptive Noise Visual
noise_x = mask_x + 9
# 画个正态分布曲线图标
x_bell = [noise_x, noise_x+1, noise_x+2, noise_x+3, noise_x+4]
y_bell = [SSD_Y+5, SSD_Y+8, SSD_Y+9, SSD_Y+8, SSD_Y+5]
# 简单的多边形模拟正态分布
poly_noise = patches.Polygon([(noise_x, SSD_Y+5), (noise_x+2, SSD_Y+9), (noise_x+4, SSD_Y+5)], fc='#f1c40f', alpha=0.6, ec='black')
ax.add_patch(poly_noise)
ax.text(noise_x+2, SSD_Y+6, r"$\mathcal{N}$", ha='center', fontsize=12)
ax.text(noise_x+2, SSD_Y+3, "Adaptive\nNoise", ha='center', fontsize=8)

# Arrow to Result
draw_arrow(ax, (noise_x+5, SSD_Y+7), (noise_x+9, SSD_Y+7))

# 4. Result (Unlearned Weights)
final_x = noise_x + 10
draw_3d_box(ax, final_x, SSD_Y+5, 4, 4, 1, '#8e44ad', r"$W'$")
ax.text(final_x+2, SSD_Y+3, "Unlearned\nWeights", ha='center', fontsize=8)


# ================= 4. 逻辑连接线 =================

# Link Fisher Result to SSD Mask (Logical connection)
# 从左边的 Fisher 矩阵连到右边的 Mask
con_fisher_mask_start = (FISHER_X+30, FISHER_Y+3)
con_fisher_mask_end = (mask_x+2, SSD_Y+5)
arrow_fm = patches.FancyArrowPatch(
    posA=con_fisher_mask_start, posB=con_fisher_mask_end, 
    arrowstyle='->', linestyle='dashed', color='#e67e22', 
    connectionstyle="arc3,rad=-0.2", linewidth=1.5
)
ax.add_patch(arrow_fm)
ax.text(50, 15, "Generate Mask\n& Scale Noise", color='#e67e22', fontsize=9, ha='center')

# Link Final Result back to Model (Update)
# 从右下角的 W' 连回上面的 Conv Block
con_update_start = (final_x+2, SSD_Y+9)
con_update_end = (45, 40) # Block 2/3 area
arrow_up = patches.FancyArrowPatch(
    posA=con_update_start, posB=con_update_end, 
    arrowstyle='->', linestyle='-', color='black', 
    connectionstyle="arc3,rad=-0.3", linewidth=2
)
ax.add_patch(arrow_up)
ax.text(80, 35, "Update Parameters", fontsize=10, fontweight='bold')


# 保存
plt.tight_layout()
save_path_png = os.path.join(SAVE_DIR, "framework_diagram.png")
save_path_pdf = os.path.join(SAVE_DIR, "framework_diagram.pdf")

plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
print(f"✅ 架构示意图已保存: {save_path_png}")
plt.show()