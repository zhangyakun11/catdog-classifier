# -*- coding: utf-8 -*-
"""训练脚本：baseline 与 improved 两个实验，全部记录到 MLflow。

用法（在项目根目录运行）：
    python ml/train.py --variant baseline --data processed --epochs 6
    python ml/train.py --variant improved --data processed --epochs 10

每个 MLflow run 记录：
- 代码版本（git commit SHA）
- 数据集版本（data/<版本>.dvc 里的 DVC md5 哈希）
- 超参数（模型结构、epochs、lr、batch_size、数据增强）
- 指标（train_loss / val_loss / val_accuracy / val_f1）
- 制品（模型权重 .pt 文件、混淆矩阵 .png）

依赖：torch / torchvision / mlflow / sklearn / matplotlib
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面后端，避免 GUI 依赖
import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 保证能 import model.py
from model import CatDogCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"
CLASSES = ["cat", "dog"]


def git_commit_sha() -> str:
    """记录当前代码版本（git commit SHA）。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def dvc_md5(data_version: str) -> str:
    """从 data/<version>.dvc 读取该数据集版本的 DVC 哈希（md5）。"""
    dvc_file = DATA_DIR / f"{data_version}.dvc"
    if dvc_file.exists():
        for line in dvc_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("md5:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


class CatDogDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None):
        self.samples = []
        for label, cls in enumerate(CLASSES):
            cls_dir = data_dir / cls
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((str(img_path), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(variant: str):
    """baseline 只做基础归一化；improved 额外加数据增强（随机翻转/旋转/颜色扰动）。"""
    base = [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if variant == "improved":
        train_tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose(base)
    val_tf = transforms.Compose(base)
    return train_tf, val_tf


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            losses.append(criterion(out, y).item())
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return acc, f1, sum(losses) / len(losses), all_labels, all_preds


def save_confusion_matrix(labels, preds, path: Path):
    # 让 matplotlib 支持中文标签（Windows 下使用微软雅黑/黑体）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="猫狗二分类训练（MLflow 追踪）")
    parser.add_argument("--variant", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--data", choices=["raw", "processed"], default="processed",
                        help="数据集版本：raw=v1，processed=v2（默认）")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}, variant={args.variant}, data={args.data}, epochs={args.epochs}")

    data_dir = DATA_DIR / args.data
    train_tf, val_tf = build_transforms(args.variant)
    dataset = CatDogDataset(data_dir, transform=None)
    n_val = max(1, int(len(dataset) * 0.2))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    # random_split 之后按子集重新应用 transform
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = CatDogCNN(variant=args.variant, input_size=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    experiment = "catdog-classifier"
    mlflow.set_experiment(experiment)
    run_name = f"{args.variant}-cnn-{args.data}"
    data_md5 = dvc_md5(args.data)

    with mlflow.start_run(run_name=run_name) as run:
        print(f"[mlflow] experiment={experiment}, run_id={run.info.run_id}, run_name={run_name}")
        # ---- 记录超参数与版本信息 ----
        mlflow.log_param("model_variant", args.variant)
        mlflow.log_param("git_commit", git_commit_sha())
        mlflow.log_param("data_version", args.data)
        mlflow.log_param("data_md5", data_md5)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("input_size", 128)
        mlflow.log_param("device", str(device))

        # ---- 训练循环，每 epoch 记录一次指标 ----
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * x.size(0)
            train_loss = train_loss_sum / len(train_ds)

            val_acc, val_f1, val_loss, labels, preds = evaluate(model, val_loader, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            print(f"  epoch {epoch:2d}/{args.epochs} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        # ---- 最终指标 ----
        val_acc, val_f1, val_loss, labels, preds = evaluate(model, val_loader, device)
        mlflow.log_metric("final_val_accuracy", val_acc)
        mlflow.log_metric("final_val_f1", val_f1)

        # ---- 记录制品：模型权重 + 混淆矩阵 ----
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACT_DIR / f"model_{args.variant}_{args.data}.pt"
        torch.save(model.state_dict(), model_path)
        cm_path = ARTIFACT_DIR / f"confusion_{args.variant}_{args.data}.png"
        save_confusion_matrix(labels, preds, cm_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(cm_path))
        print(f"[artifacts] saved -> {model_path} / {cm_path}")
        print(f"[done] run complete: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")


if __name__ == "__main__":
    main()
