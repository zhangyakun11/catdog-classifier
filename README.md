# Cat-Dog Classifier 🐱🐶

机器学习驱动的**猫狗图片二分类**应用 —— 最终课程项目（DevOps + MLOps）。

## 项目目标

- 用 PyTorch 训练/微调一个 CNN 模型，对输入的图片判断是 **猫** 还是 **狗**，并返回置信度
- 完整实践 DevOps：Git 分支策略、Docker 容器化、GitHub Actions CI/CD
- 完整实践 MLOps：DVC 数据版本管理（Azure 远程存储）、MLflow 实验追踪

## 快速开始（本地运行）

```bash
# 1. 安装依赖（torch 走 CPU 源）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. 启动 API 服务
uvicorn app.main:app --reload

# 3. 打开浏览器测试
# http://127.0.0.1:8000/docs

# 4. 用 Docker 运行（构建镜像）
docker build -t catdog-classifier .
docker run -p 8000:8000 catdog-classifier
```

## 仓库结构

```
catdog-classifier/
├── app/                  # FastAPI 应用（推理服务）
├── ml/                   # 训练/数据处理/实验文档
│   ├── model.py          # CNN 模型定义（baseline/improved）
│   ├── train.py          # 训练脚本（MLflow 追踪）
│   ├── data_pipeline.py  # 数据处理流水线（v1→v2）
│   ├── README_data.md    # 数据版本说明
│   ├── README_experiments.md  # 实验追踪说明
│   ├── registry/         # 生产模型登记
│   └── PROVENANCE.md     # 全链路溯源
├── data/                 # DVC 数据指针（原始数据在 Azure）
├── tests/                # pytest 测试（5 个，含边界情况）
├── .github/workflows/    # CI/CD 流水线
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── pyproject.toml        # ruff 代码规范配置
├── DEPLOYMENT.md         # 部署说明
└── README.md             # 本文档
```

## 分支策略与流水线（commit → build → test → staging → prod）

```
feature-*  ──PR──▶  dev  ──PR──▶  staging  ──PR──▶  main（生产）
```

| 分支 | 触发条件 | 流水线执行内容 |
|---|---|---|
| `feature-*` | 推送 / 发起 PR 到 dev/staging/main | ① 跑 **CI 质量关卡**：测试 + lint + 静态分析 + 构建 Docker 镜像 |
| `dev` | PR 合入后 | 同上（质量关卡），**CI 失败则 PR 无法合并**（分支保护） |
| `staging` | push | 质量关卡 → **自动化部署**：构建镜像并发布为 `ghcr.io/...:staging` → 冒烟测试 |
| `main` | push | 质量关卡 → **自动化部署**：构建镜像并发布为 `ghcr.io/...:latest`（生产）→ 冒烟测试 |

**质量关卡（所有分支通用）**：`pytest`（5 个测试，含边界情况）+ `ruff`（lint）+ 静态分析（compile check）+ `docker build`。任何一步失败，GitHub 分支保护规则会**阻止合并**。

**自动化部署**：镜像发布到 GitHub Container Registry（GHCR），并用刚发布的镜像启动容器做健康检查（冒烟测试）。

> 详细部署步骤与手动命令见 `DEPLOYMENT.md`。

## MLOps 组件

- **数据版本管理**：DVC + Azure Blob Storage（远程 `myazure`）。`data/raw.dvc` / `data/processed.dvc` 入库，真实数据在云端，`dvc pull` 可复现任意版本（详见 `ml/README_data.md`）
- **实验追踪**：MLflow（实验 `catdog-classifier`），每个 run 记录代码 commit、数据 md5、超参、指标、模型制品（详见 `ml/README_experiments.md`）
- **生产模型**：improved 变体（val_acc 0.675 / val_f1 0.698），登记见 `ml/registry/current_model.md`，全链路溯源见 `ml/PROVENANCE.md`

## 技术栈

PyTorch · FastAPI · DVC · MLflow · Docker · GitHub Actions · Azure Blob Storage
