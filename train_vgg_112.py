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

# ================= 升级版模型: TinyVGG (112x112 适配版) =================
class TinyVGG_112(nn.Module):
    def __init__(self, num_classes=105):
        super(TinyVGG_112, self).__init__()
        
        # Input: 112 x 112
        # Block 1: 112 -> 56
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 56 -> 28
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 28 -> 14
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4: 14 -> 7 (新增一层，提取更深层特征)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 经过4次池化：112 -> 56 -> 28 -> 14 -> 7
            # 特征图大小为 256通道 * 7 * 7
            nn.Linear(256 * 7 * 7, 1024), 
            nn.ReLU(True),
            nn.Dropout(0.2), # 降低 Dropout 比例，让模型学得更快
            nn.Linear(1024, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
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
    SAVE_PATH = os.path.join(current_dir, "checkpoint", "tiny_vgg_112.pth")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    BATCH_SIZE = 64
    EPOCHS = 25 # 更大的图，收敛会快，25轮够了
    NUM_CLASSES = 105
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using Device: {DEVICE} (Model: TinyVGG-112)")
    print(f"Dataset Path: {DATA_PATH}")

    # ================= 关键修改：112x112 分辨率 =================
    train_transform = transforms.Compose([
        transforms.Resize((120, 120)),        # 先放大
        transforms.RandomCrop((112, 112)),    # 再切 112
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_dataset = ImageFolder(root=DATA_PATH, transform=train_transform)
    if len(full_dataset) == 0: print("Error: Dataset Empty"); return

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    test_dataset.dataset.transform = test_transform 

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ================= 训练 =================
    model = TinyVGG_112(num_classes=NUM_CLASSES).to(DEVICE)
    
    # 稍微大一点的学习率 + Adam
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

    print("🚀 Start Training (112x112 Resolution)...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for inputs, labels in trainloader:
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
        
        print(f"Epoch {epoch+1}: Loss {running_loss/len(trainloader):.3f} | Train Acc {train_acc:.1f}% | Val Acc {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), SAVE_PATH)
        
        scheduler.step()

    print(f"✅ Best Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()