# 当前生产模型登记（registry/current_model.md）

本文件登记**当前部署到生产环境的模型**及其对应的代码/数据版本，作为模型注册表的轻量记录（保持与 MLflow 中的记录一致）。

## 当前生产模型

| 项目 | 值 |
|---|---|
| 模型名称 | CatDogCNN（improved 变体） |
| run_id | `5f8a440cf89a42da81707ce8fccd0ebb` |
| run 名称 | improved-cnn-processed |
| 选择日期 | 2026-10-19 |
| 状态 | ✅ 生产（Production） |

## 版本信息（代码 + 数据）

| 维度 | 值 | 说明 |
|---|---|---|
| git commit | `1a392f0` | 训练运行时的代码版本（`ml/train.py`、`ml/model.py`）；脚本最终版见 `f7fbf7c` |
| 数据版本 | v2（`data/processed`） | 清洗 + 统一缩放 128×128 的数据集 |
| 数据 DVC md5 | `13580cc951be9087fdaf268adf621a5d.dir` | 与 `data/processed.dvc` 一致 |
| 模型权重文件 | `ml/artifacts/model_improved_processed.pt` | MLflow Artifacts 中同名制品 |

## 性能指标（验证集）

| 指标 | 值 |
|---|---|
| val_accuracy | 0.6750 |
| val_f1 | 0.6977 |

## 更新记录

| 日期 | 操作 | 说明 |
|---|---|---|
| 2026-10-19 | 登记 improved 为生产模型 | 基于实验对比（F1 0.55→0.70）选定，见 `ml/README_experiments.md` |

## 说明

- 本文件是**模型注册的轻量记录**；完整的实验对比与选择理由见 [`ml/README_experiments.md`](../README_experiments.md)。
- 若后续训练出更优模型，更新本表（run_id、指标、日期）即可，保持"生产模型可追溯"。
