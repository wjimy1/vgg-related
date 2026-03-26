import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")
TARGET_CLASS = 0  # 飞机
SAVE_DIR = os.path.join(SCRIPT_DIR, "figs_algo")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

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

# ================= 数据准备 (仅 Forget Set) =================
def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    
    # 模拟场景：只有 Forget Set (Target Class)
    target_indices = [i for i, label in enumerate(trainset.targets) if label == TARGET_CLASS]
    # 我们只取 200 张作为参考梯度
    forget_set = torch.utils.data.Subset(trainset, target_indices[:200])
    forget_loader = torch.utils.data.DataLoader(forget_set, batch_size=32, shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    return forget_loader, test_loader

# ================= 1. 计算重要性 (Fisher / Gradient Sensitivity) =================
def calculate_importance(model, forget_loader):
    print("🔍 计算参数敏感度 (基于 Forget Set 梯度)...")
    importance_map = {}
    model.eval()
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    
    for inputs, labels in forget_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
    
    # 收集梯度作为重要性指标
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 使用梯度的绝对值作为敏感度
            importance_map[name] = param.grad.abs().clone()
            
    model.zero_grad()
    return importance_map

# ================= 2. 核心算法：Masked Noise Injection =================
def apply_masked_noise(model, importance_map, noise_scale, sparsity_percent=90):
    """
    params:
        noise_scale: 噪声强度 (Sigma)
        sparsity_percent: 稀疏度，例如 90 表示只攻击最敏感的前 10% 的权重
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in importance_map:
                continue
            
            # --- 策略 A: 保护浅层 ---
            # 浅层提取通用特征，动了 Retain 必掉，必须锁死
            if "features" in name:
                layer_idx = int(name.split('.')[1])
                if layer_idx < 14: # 保护前 14 层 (Conv Block 1 & 2)
                    continue

            # --- 策略 B: 稀疏攻击 ---
            imp = importance_map[name]
            # 计算该层的分位点阈值
            threshold = torch.quantile(imp, sparsity_percent / 100.0)
            
            # 生成 Mask：只针对重要性 > 阈值的权重 (Top 10%)
            mask = (imp > threshold).float()
            
            # 生成噪声并注入
            noise = torch.randn_like(param) * noise_scale
            
            # 只有被 Mask 选中的地方才加噪声
            param.add_(noise * mask)
            
    return model

# ================= 评估 =================
def evaluate(model, loader):
    model.eval()
    correct_t, total_t = 0, 0
    correct_r, total_r = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            t_mask = (labels == TARGET_CLASS)
            r_mask = (labels != TARGET_CLASS)
            
            correct_t += predicted[t_mask].eq(labels[t_mask]).sum().item()
            total_t += t_mask.sum().item()
            correct_r += predicted[r_mask].eq(labels[r_mask]).sum().item()
            total_r += r_mask.sum().item()
            
    return 100.*correct_t/(total_t+1e-8), 100.*correct_r/(total_r+1e-8)

# ================= 主程序 =================
if __name__ == "__main__":
    forget_loader, test_loader = get_dataloaders()
    
    # 1. 加载基准模型
    base_model = VGG_Small().to(DEVICE)
    base_model.load_state_dict(torch.load(MODEL_PATH))
    
    print("📊 初始状态评估...")
    orig_t, orig_r = evaluate(base_model, test_loader)
    print(f"   Original Target: {orig_t:.2f}% | Retain: {orig_r:.2f}%")
    
    # 2. 计算一次敏感度 (One-off calculation)
    imp_map = calculate_importance(base_model, forget_loader)
    
    # 3. 扫描不同的噪声强度 (Noise Sigma)
    # 我们测试一组从弱到强的噪声，看效果曲线
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    
    results_t = []
    results_r = []
    
    print("\n🚀 开始纯算法测试 (Data-Free Mode)...")
    for sigma in noise_levels:
        # 每次都从原始模型拷贝一份
        model = copy.deepcopy(base_model)
        
        # 注入噪声
        model = apply_masked_noise(model, imp_map, noise_scale=sigma, sparsity_percent=90)
        
        # 评估
        acc_t, acc_r = evaluate(model, test_loader)
        results_t.append(acc_t)
        results_r.append(acc_r)
        
        print(f"   Noise Sigma={sigma}: Target {acc_t:.2f}% | Retain {acc_r:.2f}%")
        
    # 4. 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, results_t, 'o-', color='red', label='Target Acc (Lower is better)')
    plt.plot(noise_levels, results_r, 's-', color='blue', label='Retain Acc (Higher is better)')
    plt.axhline(y=orig_r, color='gray', linestyle='--', alpha=0.5, label='Original Retain Acc')
    
    plt.xlabel('Noise Standard Deviation (Sigma)')
    plt.ylabel('Accuracy (%)')
    plt.title('Data-Free Noise Injection Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(SAVE_DIR, "datafree_pure_algo.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 曲线图已保存: {save_path}")