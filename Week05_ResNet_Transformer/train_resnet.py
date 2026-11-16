import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model_resnet import PlainCNN, ResNet18


# 读取命令行参数
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["cnn", "resnet", "no_shortcut"],
    default="cnn"
)

parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)

# 控制是否使用数据增强
parser.add_argument(
    "--augment",
    action="store_true"
)


args = parser.parse_args()

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

print("Model:", args.model)
print("Device:", device)
print("Epochs:", args.epochs)
print("Batch size:", args.batch_size)

# 普通图像转换
eval_transform = transforms.ToTensor()

# 训练数据增强
if args.augment:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
else:
    train_transform = eval_transform

# 加载训练数据
full_train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

# 验证数据不使用随机增强
full_val_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=eval_transform
)

# 测试数据不使用随机增强
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=eval_transform
)

# 保持与之前相同的数据划分
generator = torch.Generator().manual_seed(seed)

indices = torch.randperm(
    len(full_train_dataset),
    generator=generator
).tolist()

train_indices = indices[:45000]
val_indices = indices[45000:]

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)



train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# 根据命令行参数选择模型
if args.model == "cnn":
    model = PlainCNN()
elif args.model == "resnet":
    model = ResNet18(use_shortcut=True)
else:
    model = ResNet18(use_shortcut=False)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr
)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)

def run_epoch(model, loader, criterion, device, optimizer=None):
    # 判断当前是训练还是评估
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 根据模式决定是否计算梯度
    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


# 为增强实验使用独立文件名
run_name = (
    f"{args.model}_aug"
    if args.augment
    else args.model
)

history = []

best_val_accuracy = 0.0
model_path = results_dir / f"{run_name}_best.pth"

start_time = time.time()

for epoch in range(1, args.epochs + 1):

    train_loss, train_acc = run_epoch(
        model, train_loader, criterion, device, optimizer
    )

    val_loss, val_acc = run_epoch(
        model, val_loader, criterion, device
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    print(
        f"Epoch [{epoch}/{args.epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # 根据验证集准确率保存最佳模型
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), model_path)

# 保存每轮指标
history_df = pd.DataFrame(history)

history_df.to_csv(
    results_dir / f"{run_name}_history.csv",
    index=False
)

# 加载最佳模型并进行最终测试
model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)

test_loss, test_acc = run_epoch(
    model, test_loader, criterion, device
)

training_seconds = time.time() - start_time

# 保存单次实验结果
summary = pd.DataFrame([{
    "model": args.model,
    "parameters": sum(p.numel() for p in model.parameters()),
    "epochs": args.epochs,
    "best_val_accuracy": best_val_accuracy,
    "test_loss": test_loss,
    "test_accuracy": test_acc,
    "runtime_seconds": training_seconds
}])

summary.to_csv(
    results_dir / f"{run_name}_summary.csv",
    index=False
)

print("\nTraining completed.")
print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print(f"Test accuracy: {test_acc:.2%}")
print(f"Runtime: {training_seconds:.1f} seconds")
