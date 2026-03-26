import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# ================= 配置 =================
# CIFAR-10 标准均值和方差
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

BATCH_SIZE = 128
EPOCHS = 20  # 20轮足够跑到 85%+
LR = 0.01    # 使用 SGD + Momentum
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
SAVE_PATH = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")

print(f"Using Device: {DEVICE}")

# ================= 1. 数据准备 (自动下载) =================
print("正在准备数据...")

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # 经典的 CIFAR 增强
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# 自动下载到 ./data 目录
trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, # 测试集 Batch 可以大点
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ================= 2. 模型定义 (VGG-Style) =================
# 这是一个专门为器件模拟设计的纯卷积结构，没有残差干扰
class VGG_Small(nn.Module):
    def __init__(self):
        super(VGG_Small, self).__init__()
        
        # Block 1: 32x32 -> 16x16
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10), # 10类
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ================= 3. 训练循环 =================
def main():
    model = VGG_Small().to(DEVICE)
    
    # SGD with Momentum 通常在 CIFAR 上比 Adam 泛化更好
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 简单的学习率衰减: 10轮后除以10
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    print("🚀 Start Training CIFAR-10...")
    
    # 创建 checkpoint 目录
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss {running_loss/len(trainloader):.3f} | Train Acc {train_acc:.1f}% | Val Acc {test_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.4f}")
        
        scheduler.step()

    # 保存最终模型
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Training Done! Best Acc ~{test_acc:.2f}%")
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()