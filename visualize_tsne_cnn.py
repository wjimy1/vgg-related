import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os

# ================= Config =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
MODEL_PATH_ORIGINAL = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")
MODEL_PATH_UNLEARNED = os.path.join(SCRIPT_DIR, "checkpoint", "unlearned_model.pth")
FIG_DIR = os.path.join(SCRIPT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)
TARGET_CLASS = 0  # Airplane
SAMPLES = 1000

parser = argparse.ArgumentParser(description="Visualize t-SNE before/after unlearning")
parser.add_argument("--samples", type=int, default=SAMPLES, help="Number of test samples to visualize")
parser.add_argument("--target_class", type=int, default=TARGET_CLASS, help="Target class index")
parser.add_argument("--original_model", type=str, default=MODEL_PATH_ORIGINAL, help="Path to original model weights")
parser.add_argument("--unlearned_model", type=str, default=MODEL_PATH_UNLEARNED, help="Path to unlearned model weights")
parser.add_argument("--allow_simulated", action="store_true", help="Allow simulated unlearning if unlearned model is missing")
args = parser.parse_args()

TARGET_CLASS = args.target_class
SAMPLES = args.samples
MODEL_PATH_ORIGINAL = args.original_model
MODEL_PATH_UNLEARNED = args.unlearned_model

# ================= Model =================
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


def load_model_weights(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model = VGG_Small().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def build_unlearned_model():
    if os.path.exists(MODEL_PATH_UNLEARNED):
        print(f"[Info] Loaded real unlearned model: {MODEL_PATH_UNLEARNED}")
        return load_model_weights(MODEL_PATH_UNLEARNED), "real"

    if not args.allow_simulated:
        raise FileNotFoundError(
            "unlearned_model.pth not found. Generate and save a real unlearned model first, "
            "or pass --allow_simulated to explicitly use a simulated fallback."
        )

    print("[Warn] unlearned_model.pth not found, using simulated unlearning for visualization.")
    model = load_model_weights(MODEL_PATH_ORIGINAL)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "features" in name:
                param.add_(torch.randn_like(param) * 0.1)
    return model, "simulated"

# ================= Data =================
print("正在加载测试数据...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=SAMPLES, shuffle=False)
inputs, labels = next(iter(loader))
inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

print("提取原始模型特征...")
model_orig = load_model_weights(MODEL_PATH_ORIGINAL)
with torch.no_grad():
    feats_orig = model_orig.extract_features(inputs).cpu().numpy()

print("提取遗忘模型特征...")
model_unlearned, after_type = build_unlearned_model()
with torch.no_grad():
    feats_unlearned = model_unlearned.extract_features(inputs).cpu().numpy()

# ================= t-SNE =================
print("正在进行 t-SNE 降维 (这可能需要几秒钟)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)

# 拼接数据一起降维，保证空间一致性
all_feats = np.vstack([feats_orig, feats_unlearned])
all_tsne = tsne.fit_transform(all_feats)

tsne_orig = all_tsne[:SAMPLES]
tsne_unlearned = all_tsne[SAMPLES:]

# ================= Plot =================
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

plt.subplot(1, 2, 2)
title_suffix = "Real" if after_type == "real" else "Simulated"
plt.title(f"After Unlearning ({title_suffix})", fontsize=16)
plt.scatter(tsne_unlearned[~target_mask, 0], tsne_unlearned[~target_mask, 1], c='lightgray', label='Retain Classes', alpha=0.5, s=20)
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