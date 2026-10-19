# -*- coding: utf-8 -*-
"""猫狗二分类 CNN 模型定义。

两种结构：
- baseline: 简单 2 层卷积网络（无 BatchNorm/Dropout）
- improved: 3 层卷积 + BatchNorm + Dropout（更强的特征提取与正则化）

模型定义独立成模块，训练脚本 (train.py) 和 API 服务 (app/main.py) 共用，
保证训练和推理用的是同一个模型结构。
"""
import torch.nn as nn


class CatDogCNN(nn.Module):
    """猫狗二分类 CNN。

    Args:
        variant: "baseline" | "improved"
        input_size: 输入图片边长（默认 128，与 data/processed 一致）
    """

    def __init__(self, variant: str = "baseline", input_size: int = 128):
        super().__init__()
        self.variant = variant
        self.input_size = input_size

        if variant == "improved":
            # 3 层卷积 + BN + ReLU + MaxPool
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            )
            feat = 128 * (input_size // 8) ** 2
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, 2),
            )
        else:
            # baseline：2 层卷积 + ReLU + MaxPool
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            feat = 32 * (input_size // 4) ** 2
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat, 64), nn.ReLU(),
                nn.Linear(64, 2),
            )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
