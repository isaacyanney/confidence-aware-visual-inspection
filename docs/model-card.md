# Model card

## Current status

No trained model is distributed in this repository. The active adapter creates deterministic synthetic scores for API and policy testing.

## Intended future task

Single-label classification of cropped steel-surface images into six categories: crazing, inclusion, patches, pitted surface, rolled-in scale and scratches.

## Not suitable for

- automated production decisions
- worker-safety decisions
- quality acceptance or rejection
- performance or business-impact claims

## Evaluation required before model use

A trained replacement should report held-out accuracy, per-class precision and recall, a confusion matrix, calibration error and threshold behaviour. Dataset licensing, split strategy, preprocessing and checkpoint identity must be recorded.
