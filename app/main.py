"""猫狗二分类推理 API（FastAPI）。

启动服务：
    uvicorn app.main:app --reload

调用示例：
    curl -X POST -F "file=@cat.jpg" http://127.0.0.1:8000/predict

说明：
- 生产模型（improved）在模块加载时读取（import 即加载），
  不使用 FastAPI startup 事件——保证 pytest 的 TestClient 也能正常加载模型。
"""
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml"))  # 让推理代码能 import model.py

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from model import CatDogCNN  # 与训练共用同一模型定义
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

CLASSES = ["cat", "dog"]
MODEL_PATH = PROJECT_ROOT / "ml" / "artifacts" / "model_improved_processed.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 模块加载时加载生产模型（不是 startup 事件） ----
_model = CatDogCNN(variant="improved", input_size=128)
_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
_model.to(DEVICE)
_model.eval()

# 与 ml/train.py 验证阶段一致的预处理
_PREPROCESS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title="CatDog Classifier API", version="1.0.0")


def _to_tensor(image_bytes: bytes):
    """字节流 -> 模型输入张量；非图片字节会抛异常。"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _PREPROCESS(img).unsqueeze(0)


@app.get("/health")
def health():
    """健康检查：供 Docker healthcheck 与 CI 探活使用。"""
    return {"status": "ok", "model": "improved"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """接收一张图片，返回猫/狗分类与置信度。

    - 200: 正常预测
    - 400: 文件不是有效图片
    - 422: 缺少文件或文件为空
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="文件为空")

    try:
        tensor = _to_tensor(image_bytes).to(DEVICE)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="不是有效的图片文件")

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)

    pred_idx = int(probs.argmax(dim=1).item())
    confidence = float(probs.max().item())
    return {
        "prediction": CLASSES[pred_idx],
        "confidence": round(confidence, 4),
    }
