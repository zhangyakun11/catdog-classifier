# 部署说明（DEPLOYMENT.md）

本文档说明本项目的**部署方式**：本地 Docker 部署、CI/CD 自动部署，以及手动复现步骤。

## 1. 部署产物

本项目部署物是 **Docker 镜像**（含 FastAPI 推理服务 + 生产模型权重），发布到 **GitHub Container Registry（GHCR）**：

| 环境 | 镜像标签 | 触发方式 |
|---|---|---|
| 预发布（staging） | `ghcr.io/zhangyakun11/catdog-classifier:staging` | push 到 `staging` 分支自动触发 |
| 生产（prod） | `ghcr.io/zhangyakun11/catdog-classifier:latest` | push 到 `main` 分支自动触发 |

## 2. 本地部署（手动验证）

```bash
# 构建镜像
docker build -t catdog-classifier .

# 运行容器（映射 8000 端口）
docker run --rm -d --name catdog -p 8000:8000 catdog-classifier

# 健康检查
curl http://127.0.0.1:8000/health
# → {"status":"ok","model":"improved"}

# 预测
curl -X POST -F "file=@cat.jpg" http://127.0.0.1:8000/predict
# → {"prediction":"cat","confidence":0.93}

# 停止容器
docker rm -f catdog
```

## 3. CI/CD 自动部署（GitHub Actions）

流水线文件：`.github/workflows/ci.yml`

### 质量关卡（PR 到 dev/staging/main 时触发）

```
checkout → 安装依赖 → dvc pull（从 Azure 拉数据）→ ruff lint
        → 静态分析(compile) → pytest（5 个测试）→ docker build
```

配合 GitHub **分支保护**（"Require a pull request before merging" + "Require status checks"）：
**任一环节失败 → PR 无法合并**，杜绝坏代码进入受保护分支。

### 部署阶段（push 到 staging/main 时触发）

```
checkout → 登录 GHCR → docker build → docker push
        → 用发布镜像启动容器 → 冒烟测试（健康检查）→ 清理
```

### 所需 GitHub Secrets

| Secret | 用途 |
|---|---|
| `AZURE_CONNECTION_STRING` | CI 中 `dvc pull` 从 Azure 拉取数据（敏感，不入仓库） |
| `GITHUB_TOKEN` | 自动提供，用于推送 GHCR（已在 workflow 中声明 `packages: write`） |

## 4. 端到端发布流程（一次完整发版）

```bash
# ① 开发：功能分支上开发
git checkout -b feature-xxx
# ... 编码、提交 ...

# ② 合入 dev：发起 PR feature-xxx → dev
#    CI 自动跑质量关卡；通过 + 审批后合并

# ③ 合入 staging：发起 PR dev → staging
#    合并后 CI 自动部署 :staging 镜像 + 冒烟测试

# ④ 发生产：发起 PR staging → main
#    合并后 CI 自动部署 :latest 镜像 + 冒烟测试

# ⑤ 生产拉取运行
docker pull ghcr.io/zhangyakun11/catdog-classifier:latest
docker run -d -p 8000:8000 ghcr.io/zhangyakun11/catdog-classifier:latest
```

## 5. 为什么这样设计（答辩要点）

1. **多阶段晋升（dev→staging→main）**：每级都过 CI 质量关卡，越靠近生产风险越低
2. **镜像不可变**：每次部署都是新构建的镜像（tag 对应环境），回滚 = 拉取上一个镜像
3. **自动化冒烟测试**：镜像发布后立即用真实容器验证健康检查，部署失败会直接红在 CI 里
4. **敏感信息不入库**：Azure 连接字符串只在 CI 运行时从 Secrets 注入
