# Cat-Dog Classifier 🐱🐶

机器学习驱动的**猫狗图片二分类**应用 —— 最终课程项目（DevOps + MLOps）。

## 项目目标

- 用 PyTorch 训练/微调一个 CNN 模型，对输入的图片判断是 **猫** 还是 **狗**，并返回置信度
- 完整实践 DevOps：Git 分支策略、Docker 容器化、GitHub Actions CI/CD
- 完整实践 MLOps：DVC 数据版本管理（Azure 远程存储）、MLflow 实验追踪

## 快速开始（本地运行）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 API 服务
uvicorn app.main:app --reload

# 3. 打开浏览器测试
# http://127.0.0.1:8000/docs
```

## 仓库结构

```
catdog-classifier/
├── app/                  # FastAPI 应用
├── ml/                   # 训练/数据处理/实验文档
├── data/                 # DVC 数据指针（原始数据在 Azure）
├── tests/                # pytest 测试
├── .github/workflows/    # CI/CD 流水线
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── DEPLOYMENT.md         # 部署说明
└── README.md             # 本文档
```

## 分支策略

```
feature-*  →  dev  →  staging  →  main
```

- `feature-*`：功能开发分支
- `dev`：集成测试分支（受保护）
- `staging`：预发布分支（受保护，构建 staging 镜像）
- `main`：生产分支（受保护，构建生产镜像）

> 详细流程见 `DEPLOYMENT.md`。
