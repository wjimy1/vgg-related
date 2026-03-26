import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import copy
import numpy as np
from sklearn.metrics import roc_auc_score
import os

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # 加载完整数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 1. Forget Loader (用于算 Fisher)
    target_indices = [i for i, label in enumerate(trainset.targets) if label == TARGET_CLASS]
    forget_set = torch.utils.data.Subset(trainset, target_indices[:500])
    forget_loader = torch.utils.data.DataLoader(forget_set, batch_size=32, shuffle=False)
    
    # 2. Retain Loader (用于 Repair) - 过滤掉 Airplane
    retain_indices = [i for i, label in enumerate(trainset.targets) if label != TARGET_CLASS]
    retain_set = torch.utils.data.Subset(trainset, retain_indices)
    retain_loader = torch.utils.data.DataLoader(retain_set, batch_size=64, shuffle=True)

    # 3. Test Loader
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    
    return forget_loader, retain_loader, test_loader

# ================= 核心：自适应噪声 SSD =================
def apply_adaptive_ssd(model, forget_loader, dampening, selection, noise_sigma):
    print(f"⚡ 计算 Fisher 并应用自适应 SSD (Damp={dampening}, Adaptive Sigma={noise_sigma})...")
    
    fisher_dict = {}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in params.items():
        fisher_dict[n] = torch.zeros_like(p)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    for inputs, labels in forget_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for n, p in params.items():
            if p.grad is not None:
                fisher_dict[n] += p.grad.pow(2)
                
    with torch.no_grad():
        for n, p in params.items():
            f = fisher_dict[n]
            f_norm = (f - f.min()) / (f.max() - f.min() + 1e-8)
            
            # 计算抑制 Mask
            reduction = dampening * f_norm * selection * 0.1
            mask = 1.0 - reduction
            mask = torch.clamp(mask, 0.0, 1.0)
            
            # 自适应噪声 (权重变动越大，噪声越大)
            noise_scale = (1.0 - mask)
            if noise_sigma > 0:
                noise = torch.randn_like(p) * noise_sigma * noise_scale
                p.data = (p.data * mask) + noise
            else:
                p.data = p.data * mask
                
    return model

# ================= 修复函数 (新增) =================
def train_repair_one_epoch(model, retain_loader, lr=0.001):
    print("🔧 正在执行 SSD 后的快速修复 (1 Epoch)...")
    model.train()
    # 使用较小的 LR 进行微调，避免破坏遗忘效果
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    for inputs, labels in retain_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"   Repair Loss: {running_loss / len(retain_loader):.4f}")
    return model

# ================= 评估函数 =================
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
            
            target_mask = (labels == TARGET_CLASS)
            correct_target += predicted[target_mask].eq(labels[target_mask]).sum().item()
            total_target += target_mask.sum().item()
            
            retain_mask = (labels != TARGET_CLASS)
            correct_retain += predicted[retain_mask].eq(labels[retain_mask]).sum().item()
            total_retain += retain_mask.sum().item()
            
    acc_target = 100. * correct_target / (total_target + 1e-8)
    acc_retain = 100. * correct_retain / (total_retain + 1e-8)
    return acc_target, acc_retain

# ================= 主程序 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dampening', type=float, default=1.35)
    parser.add_argument('--selection', type=float, default=5.0)
    parser.add_argument('--noise_sigma', type=float, default=0.02) 
    args = parser.parse_args()

    print(f"📊 实验: SSD + Repair (Ours)")
    print(f"   Damp={args.dampening}, Sel={args.selection}, Noise={args.noise_sigma}")

    # 1. 加载
    forget_loader, retain_loader, test_loader = get_dataloaders()
    model = VGG_Small().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # 2. 初始评估
    t_acc, r_acc = evaluate(model, test_loader)
    print(f"\n[Original] Target: {t_acc:.2f}% | Retain: {r_acc:.2f}%")

    # 3. 执行 SSD (含自适应噪声)
    model = apply_adaptive_ssd(model, forget_loader, args.dampening, args.selection, args.noise_sigma)
    
    # 中间检查 (修复前)
    mid_t_acc, mid_r_acc = evaluate(model, test_loader)
    print(f"[Post-SSD] Target: {mid_t_acc:.2f}% | Retain: {mid_r_acc:.2f}% (修复前)")

    # 4. 执行快速修复 (1 Epoch)
    model = train_repair_one_epoch(model, retain_loader, lr=0.001)

    # 5. 最终评估
    final_t_acc, final_r_acc = evaluate(model, test_loader)
    
    print(f"\n----------------- 最终结果 -----------------")
    print(f"[Final]    Target: {final_t_acc:.2f}% (越低越好)")
    print(f"[Final]    Retain: {final_r_acc:.2f}% (越高越好)")
    
    if final_r_acc > 86.0 and final_t_acc < 5.0:
        print("\n🚀 成功！我们用 1 个 Epoch 就击败了 Baseline 的 5 个 Epoch！")