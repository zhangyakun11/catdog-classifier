# 实验追踪说明（README_experiments.md）

本文档记录本项目所有 MLflow 实验（run）、每个 run 使用的代码/数据版本，以及**最终选用哪个模型上生产**及原因。

## 实验环境

- **框架**：PyTorch（torch 2.14.1+cpu / torchvision 0.29.1+cpu）
- **追踪工具**：MLflow 3.17.0（实验名：`catdog-classifier`，UI 地址 `http://127.0.0.1:5000`）
- **训练数据**：`data/processed`（v2，400 张 128×128，DVC md5 `13580cc951be9087fdaf268adf621a5d.dir`）
- **训练设备**：CPU（本项目为"小数据 + 小模型 + 快速流程演示"设计，单次训练约 5–10 分钟）

## 实验列表

| run 名称 | run_id | 模型结构 | epochs | lr | batch | git commit | 数据 md5 | val_accuracy | val_f1 |
|---|---|---|---|---|---|---|---|---|---|
| **baseline-cnn-processed** | `286453e3…` | 2 层卷积（无 BN/Dropout） | 6 | 0.001 | 32 | `1a392f0` | `13580cc9…` | **0.6500** | **0.5484** |
| **improved-cnn-processed** | `5f8a440c…` | 3 层卷积 + BatchNorm + Dropout + 数据增强 | 10 | 0.001 | 32 | `1a392f0` | `13580cc9…` | **0.6750** | **0.6977** |

> run_id 完整值可在 MLflow UI（点击对应 run）或 `MlflowClient` 中查询；表格展示前 8 位便于辨认。
> `git commit` 为训练运行时的 HEAD（`1a392f0`）；随后的 DVC md5 解析修复已提交为 `f7fbf7c`，训练脚本的最终版本见该提交。

## 两个实验的差异（improved 改进了什么）

1. **网络更深**：baseline 2 层卷积 → improved 3 层卷积（32→64→128 通道），能提取更丰富的特征
2. **BatchNorm**：每层卷积后加批归一化，稳定训练、加快收敛
3. **Dropout(0.5)**：全连接层加随机失活，减少过拟合
4. **数据增强**：随机水平翻转 / 旋转 / 颜色扰动，相当于扩大了训练集（缓解 400 张小数据的过拟合）

## 选择哪个模型上生产？

**选择 improved（`5f8a440c…`）上生产**，原因：

1. **F1 分数显著更优**：0.5484 → 0.6977（提升约 0.15）。猫狗二分类正负样本均衡，但 F1 同时衡量精确率与召回率，比 accuracy 更能反映模型对"猫"和"狗"两类都分得好不好（accuracy 会掩盖某一类分得很差的情况）
2. **准确率也有提升**：0.6500 → 0.6750
3. **泛化能力更强**：数据增强 + Dropout 的设计目标就是提升泛化，F1 提升印证了这一点

**答辩要点**："为什么选 improved？—— 它的 val_f1 从 0.55 提升到 0.70，说明改进结构（更深网络 + 正则化 + 增强）确实有效，所以我们把它作为生产模型。"

## 如何复现一个实验

```bash
# baseline（6 epochs）
python ml/train.py --variant baseline --data processed --epochs 6

# improved（10 epochs）
python ml/train.py --variant improved --data processed --epochs 10

# 查看实验对比
mlflow ui   # 打开 http://127.0.0.1:5000
```

每个 run 都记录了：git commit SHA、数据集 DVC md5、全部超参数、逐 epoch 指标、模型权重 `.pt` 与混淆矩阵 `.png`（见 run 的 Artifacts 页）。

## 进一步改进方向（未实施，供答辩提问参考）

- **迁移学习**：使用 ImageNet 预训练的 ResNet18 微调（可显著提升到 90%+，但 CPU 训练较慢）
- **更多数据**：当前只用 400 张（每类 200）；原始数据集有 25,000 张可用
- **更长训练**：曲线显示 val_loss 仍在下降，可增大 epochs 或加学习率调度
