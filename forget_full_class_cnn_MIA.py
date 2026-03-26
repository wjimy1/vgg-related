import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import os

# ================= 配置 =================
parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=int, default=0, help='0=Airplane')
parser.add_argument('--dampening', type=float, default=1.0, help='SSD 阻尼强度')
parser.add_argument('--noise_sigma', type=float, default=0.0, help='器件噪声')
parser.add_argument('--selection', type=float, default=5.0, help='选择性系数')
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

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

# ================= 核心工具 =================
def get_target_loaders(target_class):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    
    # 1. Member Data (训练集中的目标类)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_idx = [i for i, label in enumerate(trainset.targets) if label == target_class]
    # 取前1000个做MIA member
    member_subset = torch.utils.data.Subset(trainset, train_idx[:1000])
    member_loader = torch.utils.data.DataLoader(member_subset, batch_size=64, shuffle=False)

    # 2. Non-Member Data (测试集中的目标类)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_idx = [i for i, label in enumerate(testset.targets) if label == target_class]
    non_member_subset = torch.utils.data.Subset(testset, test_idx)
    non_member_loader = torch.utils.data.DataLoader(non_member_subset, batch_size=64, shuffle=False)
    
    # 3. Full Test Loader (用于计算 Accuracy)
    full_test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    
    return member_loader, non_member_loader, full_test_loader

def calc_mia_score(model, member_loader, non_member_loader):
    """
    计算基于熵的 MIA AUC。
    原理：如果模型见过这些数据(Member)，预测的置信度(Entropy低)通常比没见过的(Non-Member)要高。
    """
    model.eval()
    
    def get_entropy(loader):
        entropies = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                # Entropy = -sum(p * log(p))
                e = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                entropies.extend(e.cpu().numpy())
        return entropies

    # 1. 获取熵
    member_entropy = get_entropy(member_loader)
    non_member_entropy = get_entropy(non_member_loader)
    
    # 2. 构建标签 (Member=1, Non-Member=0)
    y_true = [1] * len(member_entropy) + [0] * len(non_member_entropy)
    
    # 3. 构建分数
    # 攻击假设：熵越低 -> 越可能是 Member
    # AUC需要 score 越大代表越可能是正类(Member)。所以我们可以用 -entropy 作为 score
    y_score = -np.array(member_entropy + non_member_entropy)
    
    auc = roc_auc_score(y_true, y_score)
    return auc

def evaluate_acc(model, loader, target_class):
    model.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    target_acc = 100 * class_correct[target_class] / class_total[target_class]
    other_correct = sum(class_correct) - class_correct[target_class]
    other_total = sum(class_total) - class_total[target_class]
    retain_acc = 100 * other_correct / other_total
    return target_acc, retain_acc

def ssd_unlearn(model, loader, dampening, selection, noise_sigma):
    # 这里省略详细计算代码，直接复用你之前的逻辑
    # 为了脚本简洁，只写核心应用部分
    print("⚡ 计算 Fisher 并应用 SSD + Noise...")
    model.eval()
    fisher_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad: fisher_dict[n] = torch.zeros_like(p)
    
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        model.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad: fisher_dict[n] += p.grad.pow(2)
            
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                f = fisher_dict[n]
                f_norm = (f - f.min()) / (f.max() - f.min() + 1e-8)
                mask = 1.0 - (dampening * f_norm * selection * 0.1)
                mask = torch.clamp(mask, 0.0, 1.0)
                
                # 注入噪声
                if noise_sigma > 0:
                    noise = torch.randn_like(p) * noise_sigma
                    # 关键策略：噪声加在被修改后的权重上
                    p.data = (p.data * mask) + noise
                else:
                    p.data = p.data * mask
    return model

# ================= 主流程 =================
def main():
    print(f"📊 Experiment: Damp={args.dampening}, Noise={args.noise_sigma}")
    
    # 1. Load Model
    model = VGG_Small().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # 2. Get Data
    member_loader, non_member_loader, full_test_loader = get_target_loaders(args.target_class)
    
    # 3. Eval Baseline
    base_mia = calc_mia_score(model, member_loader, non_member_loader)
    base_tar, base_ret = evaluate_acc(model, full_test_loader, args.target_class)
    print(f"\n[Baseline] Target Acc: {base_tar:.2f}% | Retain Acc: {base_ret:.2f}% | MIA AUC: {base_mia:.4f}")
    
    # 4. Unlearn
    # 只需要一部分 Train data 来计算梯度
    forget_loader = member_loader 
    model = ssd_unlearn(model, forget_loader, args.dampening, args.selection, args.noise_sigma)
    
    # 5. Eval Unlearned
    final_mia = calc_mia_score(model, member_loader, non_member_loader)
    final_tar, final_ret = evaluate_acc(model, full_test_loader, args.target_class)
    
    print(f"\n[Final]    Target Acc: {final_tar:.2f}% | Retain Acc: {final_ret:.2f}% | MIA AUC: {final_mia:.4f}")
    
    print("\n----------------- 结论 -----------------")
    print(f"Delta MIA (Privacy Gain): {base_mia:.4f} -> {final_mia:.4f} (越接近0.5越好)")
    print(f"Delta Acc (Utility Loss): {base_ret:.2f}% -> {final_ret:.2f}% (越高越好)")

if __name__ == "__main__":
    main()