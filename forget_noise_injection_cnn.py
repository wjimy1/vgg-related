import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
import os

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")
TARGET_CLASS = 0  # Airplane

# ================= 模型定义 (VGG_Small) =================
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

    def forward(self, x):
        return self.classifier(self.features(x))

# ================= 数据准备 =================
def get_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)

    # 1. Retain Set (非 Target 的所有数据)
    retain_indices = [i for i, label in enumerate(trainset.targets) if label != TARGET_CLASS]
    retain_set = torch.utils.data.Subset(trainset, retain_indices)
    retain_loader = torch.utils.data.DataLoader(retain_set, batch_size=batch_size, shuffle=True)

    # 2. Test Loader
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    
    return retain_loader, test_loader

# ================= 核心：噪声注入逻辑 =================
def inject_noise_to_layers(model, noise_level=0.05, layer_select_ratio=0.5):
    """
    仿照你提供的脚本思想：
    1. 随机选择部分层 (Conv2d 或 Linear)
    2. 避开 BatchNorm 层
    3. 注入高斯噪声
    """
    # 收集所有可以在上面加噪声的层 (Conv 和 Linear)
    candidate_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            candidate_layers.append(module)
    
    # 随机选择一部分层
    num_to_select = int(len(candidate_layers) * layer_select_ratio)
    selected_layers = np.random.choice(candidate_layers, num_to_select, replace=False)
    
    print(f"   ⚡ Injecting noise to {len(selected_layers)}/{len(candidate_layers)} layers (Sigma={noise_level})...")
    
    with torch.no_grad():
        for layer in selected_layers:
            # 对 weight 加噪声
            if layer.weight is not None:
                noise = torch.randn_like(layer.weight) * noise_level
                layer.weight.add_(noise)
            # 对 bias 加噪声 (可选，通常影响较小)
            if layer.bias is not None:
                noise = torch.randn_like(layer.bias) * noise_level
                layer.bias.add_(noise)

# ================= 遗忘与评估流程 =================
def train_repair(model, retain_loader, optimizer, epoch):
    """
    修复训练：在噪声破坏后，用 Retain 数据把模型拉回来
    """
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(retain_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"   Epoch {epoch} Repair Loss: {running_loss / len(retain_loader):.4f}")

def evaluate(model, loader):
    model.eval()
    correct_target = 0
    total_target = 0
    correct_retain = 0
    total_retain = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Target (Class 0)
            target_mask = (labels == TARGET_CLASS)
            correct_target += predicted[target_mask].eq(labels[target_mask]).sum().item()
            total_target += target_mask.sum().item()
            
            # Retain (Class 1-9)
            retain_mask = (labels != TARGET_CLASS)
            correct_retain += predicted[retain_mask].eq(labels[retain_mask]).sum().item()
            total_retain += retain_mask.sum().item()
            
    acc_target = 100. * correct_target / (total_target + 1e-8)
    acc_retain = 100. * correct_retain / (total_retain + 1e-8)
    return acc_target, acc_retain

# ================= 主程序 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Unlearning epochs')
    parser.add_argument('--noise_level', type=float, default=0.04, help='Noise intensity (reference script uses 0.04)')
    parser.add_argument('--lr', type=float, default=0.001, help='Repair learning rate')
    args = parser.parse_args()

    print(f"🚀 开始噪声注入遗忘 (Noise Injection Unlearning)")
    print(f"   Epochs: {args.epochs} | Noise: {args.noise_level} | LR: {args.lr}")

    # 1. 数据与模型
    retain_loader, test_loader = get_dataloaders()
    model = VGG_Small().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # 2. 初始评估
    t_acc, r_acc = evaluate(model, test_loader)
    print(f"\n[Original] Target: {t_acc:.2f}% | Retain: {r_acc:.2f}%")

    # 3. 优化器 (用于修复)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # 4. 循环：注入 -> 修复
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # 步骤 A: 注入噪声 (破坏)
        # 模仿原脚本逻辑：随机挑选层，避开 BN，加入噪声
        inject_noise_to_layers(model, noise_level=args.noise_level, layer_select_ratio=0.6)
        
        # 步骤 B: 修复训练 (恢复)
        # 只用 Retain 数据训练，让模型“忘记”被破坏的 Target 知识，巩固 Retain 知识
        train_repair(model, retain_loader, optimizer, epoch)
        
        # 步骤 C: 评估
        t_acc, r_acc = evaluate(model, test_loader)
        print(f"   -> Result: Target {t_acc:.2f}% | Retain {r_acc:.2f}%")

    print("\n🏆 遗忘完成！")
    print(f"最终结果: Target {t_acc:.2f}% (越低越好), Retain {r_acc:.2f}% (越高越好)")