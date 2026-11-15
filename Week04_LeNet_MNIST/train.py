import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet


# 固定随机种子
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 设置运行参数
batch_size = 128
epochs = 5
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建输出目录
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Device:", device)
print("Epochs:", epochs)
print("Batch size:", batch_size)

# 定义图像转换
transform = transforms.ToTensor()

# 加载训练集
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# 加载测试集
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

# 创建模型
model = LeNet().to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

def train_one_epoch(model, loader, criterion, optimizer, device):
    # 切换到训练模式
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一批次的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 累计损失
        running_loss += loss.item() * images.size(0)

        # 计算预测准确率
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def evaluate(model, loader, criterion, device):
    # 切换到评估模式
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # 测试阶段不计算梯度
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / total
    test_accuracy = correct / total

    return test_loss, test_accuracy

# 保存每轮训练指标
history = []

best_accuracy = 0.0

for epoch in range(1, epochs + 1):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    # 记录当前 Epoch 的结果
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    print(
        f"Epoch [{epoch}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.4f}"
    )

    # 保存当前测试准确率最高的模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy

        torch.save(
            model.state_dict(),
            "results/lenet_best.pth"
        )

# 保存实验指标
metrics_df = pd.DataFrame(history)

metrics_df.to_csv(
    "results/metrics.csv",
    index=False
)

print("\nTraining completed.")
print(f"Best test accuracy: {best_accuracy:.4%}")
print("Metrics saved to results/metrics.csv")
print("Model saved to results/lenet_best.pth")