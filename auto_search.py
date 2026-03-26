import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import argparse
import os

# ================= 🔄 终极精细搜索空间 =================
SEARCH_SPACE = {
    # 既然 1.2 太弱，1.5 太强，我们就在中间找
    'dampening': [1.25, 1.3, 1.35, 1.4, 1.45], 
    
    # 既然 0.005 破坏力太大，我们测试更微小的噪声
    'noise_sigma': [0.0, 0.001, 0.002, 0.003, 0.004], 
    
    # 锁定 5.0，之前的实验证明它是最好的
    'selection': [5.0] 
}
# =======================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")
RESULT_CSV = os.path.join(SCRIPT_DIR, "fine_tune_results.csv")
TARGET_CLASS = 0 

# ... (其余模型定义代码保持不变，直接复用之前的即可) ...
# 为了方便，我把完整的 evaluate 和 run_ssd 逻辑再次贴在这里确保你运行无误

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

print("⏳ 正在加载数据...")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
indices = [i for i, label in enumerate(trainset.targets) if label == TARGET_CLASS]
subset = torch.utils.data.Subset(trainset, indices[:500])
forget_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

def evaluate(model):
    model.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    target_acc = 100 * class_correct[TARGET_CLASS] / class_total[TARGET_CLASS]
    retain_correct = sum(class_correct) - class_correct[TARGET_CLASS]
    retain_total = sum(class_total) - class_total[TARGET_CLASS]
    retain_acc = 100 * retain_correct / retain_total
    return target_acc, retain_acc

def run_ssd(model, dampening, noise_sigma, selection):
    fisher_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad: fisher_dict[n] = torch.zeros_like(p)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in forget_loader:
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
                if noise_sigma > 0:
                    noise = torch.randn_like(p) * noise_sigma
                    p.data = (p.data * mask) + noise
                else:
                    p.data = p.data * mask
    return model

def main():
    results = []
    total_experiments = len(SEARCH_SPACE['dampening']) * len(SEARCH_SPACE['noise_sigma']) * len(SEARCH_SPACE['selection'])
    count = 0
    print(f"\n🚀 启动精细搜索 (共 {total_experiments} 次)...")
    print(f"{'ID':<4} | {'Damp':<6} | {'Noise':<8} | {'Select':<6} | {'Target(%)':<10} | {'Retain(%)':<10} | {'Score'}")
    print("-" * 75)
    
    best_score = -999
    
    for d in SEARCH_SPACE['dampening']:
        for s in SEARCH_SPACE['selection']:
            for n in SEARCH_SPACE['noise_sigma']:
                count += 1
                model = VGG_Small().to(DEVICE)
                model.load_state_dict(torch.load(MODEL_PATH))
                model = run_ssd(model, dampening=d, noise_sigma=n, selection=s)
                tar_acc, ret_acc = evaluate(model)
                
                # 评分逻辑：我们要找 Target < 5 且 Retain 最高的
                if tar_acc > 5.0:
                    score = ret_acc - (tar_acc * 3) # 加大惩罚，强迫寻找低 Target
                else:
                    score = ret_acc 
                
                is_best = "🌟" if score > best_score and tar_acc < 5.0 else ""
                if is_best: best_score = score
                
                print(f"{count:<4} | {d:<6.2f} | {n:<8.4f} | {s:<6.1f} | {tar_acc:<10.2f} | {ret_acc:<10.2f} | {is_best}")
                
                results.append({'dampening': d, 'noise_sigma': n, 'selection': s, 'target_acc': tar_acc, 'retain_acc': ret_acc, 'score': score})

    df = pd.DataFrame(results)
    df.to_csv(RESULT_CSV, index=False)
    print(f"\n✅ 搜索结束！结果已保存至 {RESULT_CSV}")

if __name__ == "__main__":
    main()