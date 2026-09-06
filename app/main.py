from __future__ import annotations

import io
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.model import CLASSES, SyntheticModelAdapter

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
REVIEW_THRESHOLD = 0.60
ALLOWED_FORMATS = {"JPEG", "PNG"}

app = FastAPI(
    title="Confidence-Aware Visual Inspection",
    version="1.0.0",
    description="Portfolio API demonstrating explicit uncertainty handling.",
)
model = SyntheticModelAdapter()


class HealthResponse(BaseModel):
    status: str
    adapter: str
    production_model_loaded: bool


class PredictionResponse(BaseModel):
    request_id: str
    predicted_class: str
    confidence: float
    decision: str
    inference_ms: float
    scores: dict[str, float]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        adapter=model.name,
        production_model_loaded=False,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    started = time.perf_counter()
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds 5 MiB.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(status_code=415, detail="Only JPEG and PNG images are accepted.")
            image.verify()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=415, detail="The upload is not a valid image.") from exc

    result = model.predict(image_bytes)
    predicted_class, confidence = max(result.scores.items(), key=lambda item: item[1])
    if predicted_class not in CLASSES:
        raise HTTPException(status_code=500, detail="The adapter returned an unknown class.")

    return PredictionResponse(
        request_id=str(uuid.uuid4()),
        predicted_class=predicted_class,
        confidence=confidence,
        decision="classified" if confidence >= REVIEW_THRESHOLD else "review_required",
        inference_ms=round((time.perf_counter() - started) * 1000, 3),
        scores=result.scores,
    )
