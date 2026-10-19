# 数据-代码-运行全链路溯源（PROVENANCE.md）

本文件回答一个核心问题：**"当前生产模型，是怎么一步步从原始数据变出来的？"**
每一步都有版本记录（Git commit / DVC 哈希 / MLflow run），可以完整复现。

## 全链路一览

```
原始数据集（Kaggle Dogs vs Cats，微软镜像，786.7MB）
        │
        ▼  data_pipeline.py 抽样（每类 200 张，seed=42）
data/raw/   v1 ──→  DVC 纳管（data/raw.dvc）──→  Azure 远程（dvc push）
        │        commit c7d831f
        ▼  data_pipeline.py 清洗（PIL 校验）+ 统一缩放 128×128
data/processed/  v2 ──→  DVC 纳管（data/processed.dvc）──→  Azure 远程
        │        commit c7d831f
        │
        ▼  ml/train.py 训练（MLflow 追踪）
MLflow run（catdog-classifier 实验）
  ├─ baseline-cnn-processed（286453e3…）→ val_acc 0.65 / val_f1 0.55
  └─ improved-cnn-processed（5f8a440c…）→ val_acc 0.68 / val_f1 0.70  ✅ 上生产
        │
        ▼
ml/artifacts/model_improved_processed.pt（生产模型权重）
```

## 版本对照表

| 环节 | 对象 | 版本标识 | 位置 |
|---|---|---|---|
| 原始数据 | Kaggle 猫狗数据集（25,000 张） | 微软镜像 zip（`kagglecatsanddogs_5340.zip`） | 项目外（`datasets/`） |
| v1 数据 | `data/raw/`（400 张原始抽样） | DVC md5 `18b29031152d70a9bb7adb61d1264524.dir` | `data/raw.dvc` + Azure |
| v2 数据 | `data/processed/`（400 张 128×128） | DVC md5 `13580cc951be9087fdaf268adf621a5d.dir` | `data/processed.dvc` + Azure |
| 数据处理代码 | `ml/data_pipeline.py` | git `c7d831f` | 仓库 |
| 模型定义 | `ml/model.py`（baseline/improved） | git `4abf0a0` → `f7fbf7c` | 仓库 |
| 训练代码 | `ml/train.py`（MLflow 追踪） | git `4abf0a0` → `f7fbf7c` | 仓库 |
| 基线实验 | MLflow run `286453e3…` | commit `1a392f0` + 数据 md5 `13580cc9…` | MLflow |
| 改进实验（生产） | MLflow run `5f8a440c…` | commit `1a392f0` + 数据 md5 `13580cc9…` | MLflow |
| 生产模型 | `model_improved_processed.pt` | 随 improved run 的 Artifacts 记录 | `ml/artifacts/` |

## 复现路径（从零到生产模型）

```bash
# 1. 拉取代码
git clone <repo-url> && cd catdog-classifier

# 2. 拉取数据（从 Azure 远程，需 config.local 中的连接字符串）
dvc pull

# 3. 安装依赖
pip install -r requirements.txt   # torch/torchvision/mlflow/dvc/sklearn/matplotlib

# 4. 训练并复现 MLflow 记录
python ml/train.py --variant improved --data processed --epochs 10

# 5. 查看实验
mlflow ui   # http://127.0.0.1:5000
```

## 一致性说明

- MLflow 中每个 run 同时记录 `git_commit`（代码版本）与 `data_md5`（数据版本），**二者共同构成"这次实验用的什么代码 + 什么数据"的指纹**，可精确复现。
- 训练运行时的 HEAD 为 `1a392f0`；`f7fbf7c` 提交了运行后发现的 DVC md5 解析修复，不影响模型权重与指标。
- 数据版本管理由 DVC 保证：`.dvc` 指针文件入库、真实数据在 Azure，`dvc pull` 即可还原任意版本。
