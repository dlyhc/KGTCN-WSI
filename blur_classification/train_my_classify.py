import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pandas as pd

from net import DefocusedImageClassificationNet


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    best_acc = 0.0
    history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # 验证阶段
        val_loss, val_acc = test_model(model, val_loader, device, is_validation=True)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        # 记录历史
        history.append([epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc])

    # 保存历史到CSV文件
    df = pd.DataFrame(history, columns=['Epoch', 'Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
    df.to_csv('training_history.csv', index=False)


# 测试函数（用于验证）
def test_model(model, data_loader, device, is_validation=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)

    acc = correct / total
    loss = running_loss / total
    if not is_validation:
        print(f"Test Accuracy: {acc:.4f}")
    return loss, acc


# 主程序
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = DefocusedImageClassificationNet(num_classes=5).to(device)

    # 定义数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 构建数据集
    data_dir = 'classify'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}

    # 创建数据加载器
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=16) for x in
                   ['train', 'test']}
    train_loader = dataloaders['train']
    val_loader = dataloaders['test']

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=150)

    # 加载并测试最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    test_model(model, val_loader, device)