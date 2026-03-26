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
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth") # 确保路径对
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
    
    # 1. Forget Loader (只包含 Target Class 的训练数据，用于算 Fisher)
    # 取前 500 张够用了，计算快
    target_indices = [i for i, label in enumerate(trainset.targets) if label == TARGET_CLASS]
    forget_set = torch.utils.data.Subset(trainset, target_indices[:500])
    forget_loader = torch.utils.data.DataLoader(forget_set, batch_size=32, shuffle=False)
    
    # 2. Test Loaders (用于评估)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    
    # 3. Member/Non-Member Split for MIA (简化版：用部分Train和Test做MIA计算)
    # 取一部分 Train (Member) 和 Test (Non-Member)
    mia_train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(1000)), batch_size=100, shuffle=False)
    mia_test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(1000)), batch_size=100, shuffle=False)

    return forget_loader, test_loader, mia_train_loader, mia_test_loader

# ================= 核心：自适应噪声 SSD =================
def apply_adaptive_ssd(model, forget_loader, dampening, selection, noise_sigma):
    print(f"⚡ 计算 Fisher 并应用自适应 SSD (Damp={dampening}, Adaptive Sigma={noise_sigma})...")
    
    # 1. 计算 Fisher 信息
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
                
    # 2. 应用抑制 + 自适应噪声
    # 公式：Mask = 1 - (Damp * Fisher * Select)
    # 公式：Noise = Gaussian * Sigma * (1 - Mask)  <-- 只有被抑制的地方才有噪声
    
    with torch.no_grad():
        for n, p in params.items():
            f = fisher_dict[n]
            # 归一化 Fisher 到 0-1
            f_norm = (f - f.min()) / (f.max() - f.min() + 1e-8)
            
            # 计算抑制程度 (Reduction)
            reduction = dampening * f_norm * selection * 0.1
            mask = 1.0 - reduction
            mask = torch.clamp(mask, 0.0, 1.0)
            
            # --- 关键修改点：自适应噪声 ---
            # 只有当 mask < 1 (即权重被修改) 时，noise_scale 才 > 0
            # mask越小(改得越狠)，(1-mask)越大，噪声越大
            noise_scale = (1.0 - mask)
            
            if noise_sigma > 0:
                # 注意：这里的 noise_sigma 是“最大噪声强度”
                noise = torch.randn_like(p) * noise_sigma * noise_scale
                p.data = (p.data * mask) + noise
            else:
                p.data = p.data * mask
                
    return model
def train_repair_one_epoch(model, retain_loader, lr=0.001):
    print("🔧 正在执行 SSD 后的快速修复 (1 Epoch)...")
    model.train()
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

# ================= 简单的 MIA AUC 计算 =================
def compute_mia_auc(model, member_loader, non_member_loader):
    model.eval()
    confidences = []
    labels = [] # 1 for member, 0 for non-member
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Member (Train) -> 应该 Loss 小
    with torch.no_grad():
        for inputs, targets in member_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            # 使用 -Loss 作为置信度 (Loss越小越可能是Member)
            confidences.extend((-losses).cpu().numpy())
            labels.extend([1] * len(losses))
            
    # Non-Member (Test) -> 应该 Loss 大
    with torch.no_grad():
        for inputs, targets in non_member_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            confidences.extend((-losses).cpu().numpy())
            labels.extend([0] * len(losses))
            
    auc = roc_auc_score(labels, confidences)
    return auc

# ================= 主程序 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dampening', type=float, default=1.35)
    parser.add_argument('--selection', type=float, default=5.0)
    # 注意：自适应噪声的 sigma 建议比全局噪声大一些，因为会被系数缩小
    parser.add_argument('--noise_sigma', type=float, default=0.02) 
    args = parser.parse_args()

    print(f"📊 实验开始: Adaptive Noise Mode")
    print(f"   Dampening: {args.dampening}")
    print(f"   Selection: {args.selection}")
    print(f"   Max Noise: {args.noise_sigma} (Scaled by Fisher)")

    # 1. 加载数据
    forget_loader, test_loader, mia_train, mia_test = get_dataloaders()
    
    # 2. 加载模型
    model = VGG_Small().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except Exception as e:
        print(f"❌ 错误: 无法加载模型 {MODEL_PATH}。请确认路径。")
        exit()

    # 3. 基准测试
    t_acc, r_acc = evaluate(model, test_loader)
    print(f"\n[Original] Target: {t_acc:.2f}% | Retain: {r_acc:.2f}%")

    # 4. 执行遗忘
    model = apply_adaptive_ssd(model, forget_loader, args.dampening, args.selection, args.noise_sigma)
    
    # 5. 最终测试
    final_t_acc, final_r_acc = evaluate(model, test_loader)
    mia_score = compute_mia_auc(model, mia_train, mia_test)
    
    print(f"\n[Adaptive] Target: {final_t_acc:.2f}% | Retain: {final_r_acc:.2f}% | MIA AUC: {mia_score:.4f}")
    
    print("\n----------------- 结果分析 -----------------")
    print(f"Target Drop: {t_acc:.2f}% -> {final_t_acc:.2f}% (越低越好)")
    print(f"Retain Drop: {r_acc:.2f}% -> {final_r_acc:.2f}% (越高越好)")
    print(f"Privacy: MIA AUC = {mia_score:.4f} (越接近0.5越安全)")
    
    # 简单的自动评价
    if final_t_acc < 10.0 and final_r_acc > 77.0:
        print("🏆 结论: 这种方法不仅忘得干净，而且比全局噪声保留了更多通用能力！")
    elif final_r_acc > 77.0:
        print("✅ 结论: 通用能力保护得很好，Retain Acc 很高。")