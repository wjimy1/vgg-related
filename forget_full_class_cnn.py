import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import argparse
import os

# ================= 配置参数 =================
# 你可以通过命令行修改这些参数
parser = argparse.ArgumentParser(description='SSD Full Class Unlearning with Device Noise')
parser.add_argument('--target_class', type=int, default=0, help='要遗忘的类别索引 (0=Airplane)')
parser.add_argument('--dampening', type=float, default=10.0, help='SSD 阻尼强度 (Lambda), 值越大忘得越狠')
parser.add_argument('--noise_sigma', type=float, default=0.0, help='器件噪声标准差 (模拟 C2C 变化), 0 表示无噪声')
parser.add_argument('--selection', type=float, default=10.0, help='选择性系数 (Alpha), 放大重要权重的差异')
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ================= 模型定义 (必须与训练一致) =================
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

# ================= 数据加载 =================
def get_loaders(target_class):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    
    # 1. 获取训练集 (用于计算 Fisher Info)
    # 我们只挑选属于 "目标类别" 的图片，因为我们要找的是对"这个类别"重要的突触
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 筛选出 Target Class 的索引
    indices = [i for i, label in enumerate(trainset.targets) if label == target_class]
    # 取一部分样本即可 (500张足够计算梯度)
    subset = torch.utils.data.Subset(trainset, indices[:500])
    forget_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)
    
    # 2. 获取测试集 (用于评估)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    
    return forget_loader, test_loader

# ================= 评估函数 =================
def evaluate(model, loader, target_class):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # 计算指标
    target_acc = 100 * class_correct[target_class] / class_total[target_class]
    
    # 计算保留类别的平均准确率 (Retain Accuracy)
    retain_correct = sum(class_correct) - class_correct[target_class]
    retain_total = sum(class_total) - class_total[target_class]
    retain_acc = 100 * retain_correct / retain_total
    
    return target_acc, retain_acc

# ================= 核心：带噪声的 SSD =================
def ssd_unlearn(model, forget_loader, dampening, selection_weight, noise_sigma):
    print(f"\n⚡ [Step 1] 计算 Fisher Information (针对类别 {args.target_class})...")
    
    # 初始化 Fisher 字典
    fisher_dict = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            fisher_dict[name] = torch.zeros_like(p)

    model.eval()
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    
    # 累积梯度
    for inputs, labels in forget_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher_dict[name] += p.grad.pow(2) # 梯度平方
        model.zero_grad()
        
    print(f"⚡ [Step 2] 应用 SSD 阻尼 + 器件噪声 (Sigma={noise_sigma})...")
    
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                fisher = fisher_dict[name]
                # 归一化 Fisher 到 [0, 1]
                # 加上 epsilon 防止除零
                fisher_norm = (fisher - fisher.min()) / (fisher.max() - fisher.min() + 1e-8)
                
                # ==== SSD 核心公式 ====
                # 只有 Fisher 值大的 (对目标类重要的)，才会被阻尼
                # Mask = 1 - lambda * Fisher^alpha
                # 这里的 selection_weight 类似 alpha，拉大重要性的差距
                damping_factor = dampening * fisher_norm
                
                # 简单的线性阻尼 (你也可以尝试指数级)
                mask = 1.0 - (damping_factor * selection_weight * 0.1)
                mask = torch.clamp(mask, min=0.0, max=1.0) # 保证不反转符号
                
                # 应用阻尼
                new_weight = p.data * mask
                
                # ==== 创新点：注入器件噪声 ====
                if noise_sigma > 0:
                    # 模拟忆阻器编程时的随机性: W_final ~ N(W_target, sigma^2)
                    noise = torch.randn_like(p.data) * noise_sigma
                    # 你可以选择：噪声是加在所有权重上？还是只加在被修改的权重上？
                    # 硬件上通常是对整个阵列操作会有读写噪声，这里假设全局注入
                    new_weight = new_weight + noise
                
                p.data.copy_(new_weight)
                
    return model

# ================= 主程序 =================
def main():
    print(f"🎯 任务：Full Class Unlearning | Target Class: {args.target_class} ({classes[args.target_class]})")
    print(f"⚙️ 参数：Dampening={args.dampening}, Selection={args.selection}, Noise Sigma={args.noise_sigma}")

    # 1. 加载模型
    model = VGG_Small().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except:
        print("❌ 错误：找不到模型，请先运行 train_cifar10.py")
        return

    forget_loader, test_loader = get_loaders(args.target_class)

    # 2. 评估 Baseline
    print("\n📊 评估原始模型...")
    orig_target, orig_retain = evaluate(model, test_loader, args.target_class)
    print(f"   >>> Target Acc: {orig_target:.2f}% | Retain Acc: {orig_retain:.2f}%")

    # 3. 执行遗忘
    model = ssd_unlearn(model, forget_loader, args.dampening, args.selection, args.noise_sigma)

    # 4. 评估遗忘后
    print("\n📊 评估遗忘后模型...")
    final_target, final_retain = evaluate(model, test_loader, args.target_class)
    print(f"   >>> Target Acc: {final_target:.2f}% | Retain Acc: {final_retain:.2f}%")

    # 5. 结论
    print("\n📝 实验报告:")
    print(f"Target Class Drop: {orig_target:.2f}% -> {final_target:.2f}% (越低越好)")
    print(f"Retained Stability: {orig_retain:.2f}% -> {final_retain:.2f}% (越高越好)")

if __name__ == "__main__":
    main()