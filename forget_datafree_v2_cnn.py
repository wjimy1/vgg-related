import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import copy
import numpy as np
import os

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")
TARGET_CLASS = 0  # 飞机

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
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 只取 200 张飞机图片计算梯度
    target_indices = [i for i, label in enumerate(trainset.targets) if label == TARGET_CLASS]
    forget_set = torch.utils.data.Subset(trainset, target_indices[:200])
    forget_loader = torch.utils.data.DataLoader(forget_set, batch_size=32, shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    return forget_loader, test_loader

# ================= 1. 计算梯度敏感度 =================
def calculate_sensitivity(model, forget_loader):
    print("🔍 [Sensitivity] 正在定位高敏感权重...")
    sensitivity_map = {}
    model.eval()
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    
    for inputs, labels in forget_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            sensitivity_map[name] = param.grad.abs().clone()
            
    model.zero_grad()
    return sensitivity_map

# ================= 2. 核心：精准噪声注入 (V2) =================
def apply_surgical_noise(model, sensitivity_map, noise_scale, sparsity_percent=99.0):
    """
    改进点：
    1. 只攻击 classifier 层
    2. 使用 Top 1% (sparsity=99)
    3. 使用乘性噪声 (Multiplicative)
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in sensitivity_map:
                continue
            
            # 【策略修正】绝对不动特征提取层 (Features)
            # 因为没有 Retain Data，一旦破坏特征提取器，神仙也救不回来
            if "features" in name:
                continue

            # 只处理 classifier (Linear layers)
            sens = sensitivity_map[name]
            
            # 计算 Top 1% 阈值
            threshold = torch.quantile(sens, sparsity_percent / 100.0)
            mask = (sens > threshold).float()
            
            if mask.sum() == 0:
                continue

            # 【噪声修正】乘性噪声： p_new = p_old * (1 + noise)
            # 这样保证了权重的符号不变，且噪声大小与权重幅值成正比
            noise = torch.randn_like(param) * noise_scale
            perturbation = param * noise * mask
            
            param.add_(perturbation)
            
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
    base_model = VGG_Small().to(DEVICE)
    base_model.load_state_dict(torch.load(MODEL_PATH))
    
    print("\n📊 [Baseline] 原始模型性能")
    orig_t, orig_r = evaluate(base_model, test_loader)
    print(f"   Target: {orig_t:.2f}% | Retain: {orig_r:.2f}%")
    
    # 1. 定位
    sens_map = calculate_sensitivity(base_model, forget_loader)
    
    # 2. 扫描噪声 (因为是乘性噪声，Scale 可以大一点)
    noise_levels = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    print("\n🚀 [Experiment] Data-Free Surgical Noise Injection")
    print(f"   Strategy: Only Classifier Layers | Top 1% Weights | Multiplicative Noise")
    print("-" * 60)
    
    for sigma in noise_levels:
        model = copy.deepcopy(base_model)
        model = apply_surgical_noise(model, sens_map, noise_scale=sigma, sparsity_percent=99.0)
        
        t_acc, r_acc = evaluate(model, test_loader)
        
        # 简单计算一下 Trade-off 分数
        print(f"   Sigma={sigma:<4}: Target {t_acc:6.2f}% | Retain {r_acc:6.2f}%")

    print("-" * 60)
    print("💡 分析: 理想情况是 Target 降至 <10%，同时 Retain 保持 >80%。")