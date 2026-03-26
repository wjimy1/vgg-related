import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# ================= 路径修复 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# ================= 定义加强版模型: TinyVGG =================
class TinyVGG(nn.Module):
    def __init__(self, num_classes=105):
        super(TinyVGG, self).__init__()
        
        # Block 1: 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), # 8192 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.4), #稍微降低 dropout
            nn.Linear(512, num_classes)
        )

        # 【关键】权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        print("💡 Applying Kaiming Initialization...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

def main():
    # ================= 配置 =================
    data_candidates = [
        os.path.join(current_dir, "105_classes_pins_dataset"),
        os.path.join(current_dir, "..", "selective-synaptic-dampening-main", "selective-synaptic-dampening-main", "src", "105_classes_pins_dataset"),
        "105_classes_pins_dataset",
    ]
    DATA_PATH = next((p for p in data_candidates if os.path.exists(p)), data_candidates[0])
    SAVE_PATH = os.path.join(current_dir, "checkpoint", "tiny_vgg_pins.pth")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    BATCH_SIZE = 64
    EPOCHS = 30
    NUM_CLASSES = 105
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using Device: {DEVICE}")
    print(f"Dataset Path: {DATA_PATH}")

   # ================= 修改后的数据增强 =================
    # 既然追求速度和准确率，先去掉干扰项，让模型看清楚全脸
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 直接缩放到 64x64，不切了！
        transforms.RandomHorizontalFlip(), # 左右翻转是可以的
        transforms.ColorJitter(brightness=0.2, contrast=0.2), #稍微调点颜色，增加鲁棒性
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 验证集保持不变
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_dataset = ImageFolder(root=DATA_PATH, transform=train_transform)
    # 确保数据集不为空
    if len(full_dataset) == 0:
        print("❌ Error: Dataset empty!")
        return

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    test_dataset.dataset.transform = test_transform 

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # ================= 模型与训练 =================
    model = TinyVGG(num_classes=NUM_CLASSES).to(DEVICE)
    
    # 使用 Adam, LR=0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
   # 原来是 step_size=10，这导致第 20 轮时 LR 已经很小了
    # 我们改成 step_size=20，让它在大 LR 下多跑一会儿
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    print("🚀 Start Training TinyVGG...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
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
            
            # 每 50 个 Batch 打印一次，确保它在动
            if (i+1) % 50 == 0:
                print(f"Step [{i+1}/{len(trainloader)}] Loss: {loss.item():.4f} | Batch Acc: {100.*predicted.eq(labels).sum().item()/labels.size(0):.1f}%")

        epoch_loss = running_loss / len(trainloader)
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
        
        print(f"✅ Epoch {epoch+1} Summary: Train Acc {train_acc:.2f}% | Val Acc {test_acc:.2f}% | Loss {epoch_loss:.4f}")
        
        # 只要 Val Acc 超过 10% 就保存，防止后面崩了没东西用
        if test_acc > 10.0:
            torch.save(model.state_dict(), SAVE_PATH)
        
        scheduler.step()

if __name__ == "__main__":
    main()