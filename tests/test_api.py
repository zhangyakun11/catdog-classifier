# -*- coding: utf-8 -*-
"""FastAPI 推理接口的单元测试（pytest）。

覆盖场景（含边界情况）：
1. 正常图片 → 返回预测标签 + 置信度
2. 非图片文件（文本）→ 400
3. 空文件 → 422
4. 缺少文件字段 → 422
5. 健康检查端点 → 200
"""
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# 用训练数据中的一张真实猫图作为正常图片用例（数据由 DVC 管理）
CAT_IMG = Path(__file__).resolve().parent.parent / "data" / "processed" / "cat"


def _first_cat_image():
    images = sorted(CAT_IMG.glob("*.jpg"))
    assert images, "测试需要 data/processed/cat 下至少一张图片（先运行 dvc pull 或数据处理）"
    return images[0]


def test_predict_returns_label_and_confidence():
    """边界外：正常输入——应返回合法标签与 0~1 的置信度。"""
    img = _first_cat_image()
    with img.open("rb") as f:
        resp = client.post(
            "/predict",
            files={"file": ("cat.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in ("cat", "dog")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_rejects_non_image():
    """边界：非图片文件（文本）应返回 400。"""
    resp = client.post(
        "/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert resp.status_code == 400
    assert "不是有效的图片文件" in resp.json()["detail"]


def test_predict_rejects_empty_file():
    """边界：空文件（0 字节）应返回 422。"""
    resp = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 422


def test_predict_requires_file_field():
    """边界：请求完全不带文件字段应返回 422（FastAPI 校验）。"""
    resp = client.post("/predict")
    assert resp.status_code == 422


def test_health_endpoint():
    """健康检查应返回 200 与模型信息。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
