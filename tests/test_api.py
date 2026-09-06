from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (16, 16), color=(64, 96, 128)).save(stream, format="PNG")
    return stream.getvalue()


def test_health_discloses_synthetic_adapter() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "adapter": "synthetic-deterministic-v1",
        "production_model_loaded": False,
    }


def test_prediction_contract_and_confidence_boundary() -> None:
    response = client.post("/predict", files={"file": ("sample.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] in payload["scores"]
    assert payload["decision"] in {"classified", "review_required"}
    assert payload["decision"] == (
        "classified" if payload["confidence"] >= 0.60 else "review_required"
    )
    assert len(payload["scores"]) == 6
    assert payload["request_id"]
    assert payload["inference_ms"] >= 0


def test_invalid_file_is_rejected() -> None:
    response = client.post("/predict", files={"file": ("sample.txt", b"not an image", "text/plain")})
    assert response.status_code == 415
