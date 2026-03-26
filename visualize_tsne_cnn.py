import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import os

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_ORIGINAL = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")
FIG_DIR = os.path.join(SCRIPT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)
# 注意：这里需要你刚才跑出的那个最佳模型，如果你没保存，可以用代码里的逻辑重新加载权重
# 假设你现在的内存里没有模型，我们演示时对比“原始模型”和“加噪后的逻辑”
TARGET_CLASS = 0  # Airplane
SAMPLES = 1000    # 采样点数，太多跑得慢

# ================= 模型定义 =================
class VGG_Small(nn.Module):
    def __init__(self):
        super(VGG_Small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    # 修改 forward，提取特征层输出
    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x 

# ================= 准备数据 =================
print("正在加载测试数据...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=SAMPLES, shuffle=True)
inputs, labels = next(iter(loader))
inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

# ================= 核心：复现你的 Unlearning 操作 =================
def get_unlearned_model(dampening=1.35, noise_sigma=0.003, selection=5.0):
    model = VGG_Small().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH_ORIGINAL))
    
    # 重新计算 Fisher 并应用 (简化版逻辑，复用你之前的参数)
    # 为了演示，这里我们需要快速模拟一下那个 mask 操作
    # 实际画图时，最好是 load 你保存好的 unlearned_model.pth
    # 这里我们演示“假设模型已经应用了权重”
    
    # ⚠️ 注意：为了看到真实效果，建议你直接在之前的脚本里加一行 torch.save(model.state_dict(), 'unlearned_model.pth')
    # 如果没有保存，我们这里只能对比“原始模型”和“随机初始化模型”(作为极端对比)
    # 或者你需要把 calculate_fisher 的代码搬过来跑一遍
    
    return model

# 加载原始模型
print("提取原始模型特征...")
model_orig = VGG_Small().to(DEVICE)
model_orig.load_state_dict(torch.load(MODEL_PATH_ORIGINAL))
model_orig.eval()
with torch.no_grad():
    feats_orig = model_orig.extract_features(inputs).cpu().numpy()

# ==========================================
# ⚡ 关键：这里需要你的 Unlearned Model
# 如果你之前没保存，请先用 forget_full_class.py 保存一个 .pth 文件
# 这里为了演示效果，我用一个临时技巧：手动给 Airplane 相关的权重加噪声模拟效果
# ==========================================
print("提取遗忘模型特征 (模拟)...")
model_unlearned = VGG_Small().to(DEVICE)
model_unlearned.load_state_dict(torch.load(MODEL_PATH_ORIGINAL))

# --- 模拟遗忘效果 (你在实际使用时，请加载真实的 unlearned_model.pth) ---
# 这里我们简单加点强噪声来模拟 t-SNE 的散开效果，方便你调试代码
with torch.no_grad():
    for name, param in model_unlearned.named_parameters():
        if "features" in name: # 破坏特征提取层
             param.add_(torch.randn_like(param) * 0.1) 
# -------------------------------------------------------------------

model_unlearned.eval()
with torch.no_grad():
    feats_unlearned = model_unlearned.extract_features(inputs).cpu().numpy()

# ================= t-SNE 降维 =================
print("正在进行 t-SNE 降维 (这可能需要几秒钟)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)

# 拼接数据一起降维，保证空间一致性
all_feats = np.vstack([feats_orig, feats_unlearned])
all_tsne = tsne.fit_transform(all_feats)

tsne_orig = all_tsne[:SAMPLES]
tsne_unlearned = all_tsne[SAMPLES:]

# ================= 绘图 =================
labels_np = labels.cpu().numpy()
target_mask = (labels_np == TARGET_CLASS)

plt.figure(figsize=(16, 7))

# 图1：原始模型
plt.subplot(1, 2, 1)
plt.title("Before Unlearning (Original)", fontsize=16)
# 画非目标类 (灰色背景)
plt.scatter(tsne_orig[~target_mask, 0], tsne_orig[~target_mask, 1], c='lightgray', label='Retain Classes', alpha=0.5, s=20)
# 画目标类 (红色高亮)
plt.scatter(tsne_orig[target_mask, 0], tsne_orig[target_mask, 1], c='red', label='Target (Airplane)', alpha=0.8, s=40, edgecolors='k')
plt.legend()
plt.xticks([])
plt.yticks([])

# 图2：遗忘模型
plt.subplot(1, 2, 2)
plt.title("After Unlearning (Ours)", fontsize=16)
# 画非目标类
plt.scatter(tsne_unlearned[~target_mask, 0], tsne_unlearned[~target_mask, 1], c='lightgray', label='Retain Classes', alpha=0.5, s=20)
# 画目标类
plt.scatter(tsne_unlearned[target_mask, 0], tsne_unlearned[target_mask, 1], c='blue', label='Target (Airplane)', alpha=0.8, s=40, edgecolors='k')
plt.legend()
plt.xticks([])
plt.yticks([])

plt.tight_layout()
output_png = os.path.join(FIG_DIR, "tsne_visualization.png")
output_svg = os.path.join(FIG_DIR, "tsne_visualization.svg")
plt.savefig(output_png, dpi=300)
plt.savefig(output_svg)
print(f"✅ 图表已保存为 {output_png} 和 {output_svg}")
plt.show()