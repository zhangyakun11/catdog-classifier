# 数据版本说明（README_data.md）

本文档记录本项目训练数据集的**来源**、**版本变化**以及**与代码提交（commit）的对应关系**。

## 数据来源

- **数据集**：Kaggle 竞赛 **Dogs vs Cats**（经典猫狗二分类数据集）
- **获取方式**：微软官方镜像下载（可直连，无需 Kaggle 账号）
  - 下载地址：`https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip`
  - 压缩包大小：约 786.7 MB，解压后含 `PetImages/Cat/`（约 12,500 张）和 `PetImages/Dog/`（约 12,500 张）
- **使用范围**：本项目仅抽取其中**每类 200 张**（共 400 张）用于训练与版本管理演示（project 明确"数据集小（玩具规模）也可以，重点是流程"）

## 数据处理流程

数据流水线脚本：`ml/data_pipeline.py`

```bash
python ml/data_pipeline.py --source <PetImages目录> --n 200 --size 128
```

该脚本完成两步：
1. **v1 生成**：从原始数据每类随机抽样 200 张，原样复制到 `data/raw/`（保持原始尺寸，`random.seed(42)` 保证可复现）
2. **v2 生成**：对 v1 逐张校验完整性（`PIL verify()`），丢弃损坏/非 JPEG 图片，并统一缩放为 `128x128`，保存到 `data/processed/`

## 数据集版本

| 版本 | 目录 | 内容 | 张数 | 用途 |
|---|---|---|---|---|
| **v1** | `data/raw/` | 原始抽样（保持原始尺寸，可能含损坏图） | 400（cat 200 + dog 200） | 基线训练用数据集 |
| **v2** | `data/processed/` | 清洗（校验完整性）+ 统一缩放 128×128 | 400（cat 200 + dog 200） | 改进训练用数据集（推荐） |

> 注：本项目抽样批次恰好全部图片完好（v2 丢弃 0 张损坏）；清洗逻辑（损坏检测）已内置于 `data_pipeline.py`，可复用于更大规模抽样。

## 代码提交（commit）与数据集版本对应关系

| Git commit（SHA） | 说明 | 关联的数据集版本 |
|---|---|---|
| `c7d831f` | 添加 DVC 数据版本管理（v1 raw + v2 processed） | v1（`data/raw.dvc`）与 v2（`data/processed.dvc`）的**首次纳管** |
| `43c8bb1` | 配置 Azure 远程存储 | 数据可推送/拉取到 Azure（`azure://dvc`） |

DVC 哈希（数据指纹）：
- v1（`data/raw.dvc` → `md5`）：`18b29031152d70a9bb7adb61d1264524.dir`
- v2（`data/processed.dvc` → `md5`）：`13580cc951be9087fdaf268adf621a5d.dir`

**对应关系说明**：训练脚本 `ml/train.py` 会把当前代码的 `git commit SHA` 与所用数据版本的 `DVC md5` 一并记录到 MLflow（见 `README_experiments.md`），因此可以回答："**这个模型是由哪个确切的数据集版本产生的？**"

## 数据存储位置

- **Git 仓库**：只保存 `.dvc` 指针文件（`data/raw.dvc`、`data/processed.dvc`），**原始图片不进入 Git**（由 `.gitignore` 拦截）
- **远程存储**：Azure Blob Storage（账号 `catdogdvc2026`，容器 `dvc`），通过 `dvc push` 上传

```bash
dvc remote -v                 # 查看远程配置
dvc push                      # 推送数据到 Azure
dvc pull                      # 从 Azure 拉取数据（其他机器复现时用）
```
