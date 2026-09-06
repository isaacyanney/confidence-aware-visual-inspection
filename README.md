# Confidence-Aware Visual Inspection

An independent portfolio project for classifying six steel-surface defect categories while making uncertainty visible at the API boundary.

The repository focuses on the parts that matter in a review: a clear inference contract, confidence thresholds, request tracing, safe failure behaviour, container packaging and automated tests. A real PyTorch CNN is trained and evaluated in automation on procedurally generated textures. The API deliberately keeps a deterministic synthetic adapter as its default.

## Defect classes

- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

## What is demonstrated

- FastAPI endpoints for health checks and predictions
- confidence-aware outcomes: classified, review_required or rejected
- request IDs and inference timing
- image validation and bounded upload size
- reproducible PyTorch training on procedurally generated textures
- held-out accuracy, per-class precision/recall, confusion matrix and calibration error
- downloadable checkpoint and metrics from each successful training run
- Docker packaging and GitHub Actions validation
- tests covering healthy, uncertain and invalid requests

## Evidence boundary

This is an independent learning and portfolio project. It does not claim factory deployment, live production use, business impact or ownership of an industrial inspection system. Training uses procedurally generated textures, not real industrial images. Workflow artifacts are experimental evidence only.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API interface.

## Test

```bash
pytest
```

## Train and evaluate

```bash
pip install -e ".[ml]"
python -m ml.train --epochs 5 --samples-per-class 64
```

The command writes `metrics.json` and a PyTorch checkpoint under `artifacts/`.
The report identifies the dataset as synthetic and sets
`production_evidence` to `false`.

## API contract

`GET /health` reports readiness and the active adapter.

`POST /predict` accepts a JPEG or PNG. A response includes:

- `request_id`
- `predicted_class`
- `confidence`
- `decision`
- `inference_ms`
- scores for all six classes

The service never presents a low-confidence result as certain. Scores below the configured review threshold return `review_required`.

## Repository map

```text
app/                 API and model adapter
tests/               behavioural tests
docs/                architecture and model-card notes
.github/workflows/   automated validation
```

## Next technical improvements

1. Replace procedural textures with a properly licensed real-image dataset.
2. Compare the small CNN with a transfer-learning baseline.
3. Version the checkpoint and preprocessing contract together.
4. Add drift monitoring before considering any operational use.
