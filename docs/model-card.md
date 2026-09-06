# Model card

## Current status

The automated workflow trains and evaluates a small PyTorch CNN on procedurally
generated textures and publishes its checkpoint and metrics as a workflow
artifact. The API keeps a deterministic synthetic adapter as its default so no
experimental checkpoint can be mistaken for a production model.

## Intended future task

Single-label classification of cropped steel-surface images into six categories: crazing, inclusion, patches, pitted surface, rolled-in scale and scratches.

## Not suitable for

- automated production decisions
- worker-safety decisions
- quality acceptance or rejection
- performance or business-impact claims

## Evaluation evidence

The training workflow records held-out synthetic accuracy, per-class precision
and recall, a confusion matrix and expected calibration error. These results
only demonstrate that the software pipeline works on the generated patterns.
They do not measure performance on real steel images.

## Evaluation required before real-image use

A trained replacement should report held-out accuracy, per-class precision and recall, a confusion matrix, calibration error and threshold behaviour. Dataset licensing, split strategy, preprocessing and checkpoint identity must be recorded.
