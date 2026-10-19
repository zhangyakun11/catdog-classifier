# 猫狗二分类推理 API —— 精简镜像（Dockerfile）
#
# 设计要点（精简 + 安全）：
# 1) 只用运行时依赖，不装训练/MLflow 相关包
# 2) 利用层缓存：依赖清单先复制安装，代码后复制
# 3) 非 root 用户运行（安全实践）
# 4) 自带 HEALTHCHECK 健康检查

FROM python:3.12-slim

WORKDIR /app

# ---- 第 1 层：安装依赖（先用 COPY requirements 触发层缓存）----
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ---- 第 2 层：复制应用代码 + 模型定义 + 生产模型权重 ----
COPY app/ ./app/
COPY ml/model.py ./ml/model.py
COPY ml/artifacts/model_improved_processed.pt ./ml/artifacts/model_improved_processed.pt

# ---- 第 3 层：安全加固 ----
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查（slim 镜像无 curl，用 python 探测）
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
