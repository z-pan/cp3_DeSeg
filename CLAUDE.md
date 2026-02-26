# Project: Cellpose 3 Training & Testing Pipeline

## ⚠️ CRITICAL: This project targets Cellpose 3 (cyto3), NOT Cellpose-SAM (cpsam)

### Architecture Distinction
- **Cellpose 3** uses the Res-U-Net architecture (backbone="default"),
  pretrained_model="cyto3"
- **Cellpose-SAM** uses the SegFormer/SAM architecture (backbone="transformer"),
  pretrained_model="cpsam"
- In cellpose >= 4.0, the DEFAULT model is cpsam. You MUST explicitly set
  pretrained_model="cyto3" when initializing models.
- The old `models.Cellpose` class has been removed in cellpose 4.
  Use `models.CellposeModel(pretrained_model="cyto3")`.

### Cellpose 3 Capabilities (both must be implemented)
1. **Segmentation**: cell/nucleus segmentation using cyto3 model
   - API: `models.CellposeModel(pretrained_model="cyto3")`
   - Training: `train.train_seg()`
   - Requires: channels parameter, diameter parameter
2. **Image Restoration**: denoise / deblur / upsample degraded images
   - API: `denoise.CellposeDenoiseModel(model_type="cyto3",
     restore_type="denoise_cyto3")`
   - Training: `denoise.DenoiseModel.train()`
   - Restore types: denoise_cyto3, deblur_cyto3, upsample_cyto3
   - Training uses chained architecture: trainable restoration network →
     frozen segmentation network, trained with segmentation loss +
     perceptual loss

### What NOT to do
- Do NOT use pretrained_model="cpsam" — wrong architecture
- Do NOT omit the channels parameter — cyto3 requires it
- Do NOT omit the diameter parameter — cyto3 is size-dependent
- Do NOT use the removed `models.Cellpose` class
- Do NOT assume channel-invariance or size-invariance

## Tech Stack
- Python 3.10+
- PyTorch (GPU-enabled)
- cellpose >= 3.0 (Cellpose 3 with restoration support)
  NOTE: if using cellpose 4.x, cyto3 is still available but not default
- Standard scientific Python: numpy, scikit-image, matplotlib, tifffile

## Project Structure Conventions
- `configs/` — YAML config files for experiments
- `src/` — Core pipeline modules
- `scripts/` — Entry point scripts (train.py, test.py, evaluate.py)
- `notebooks/` — Jupyter notebooks for visualization and analysis
- `data/` — Symlinked or .gitignored data directory
- `outputs/` — Training outputs (models, logs, predictions)

## Code Style
- Type hints on all function signatures
- Docstrings (Google style) on all public functions
- Use argparse or hydra for CLI arguments
- Use logging module, not print statements
- Config-driven: all hyperparameters in YAML, not hardcoded
```
- `configs/experiment/train_deblur.yaml` — deblur restoration training
- `configs/experiment/train_upsample.yaml` — upsample restoration training
- `configs/experiment/restore_then_seg.yaml` — chained pipeline

Do NOT create any config with pretrained_model="cpsam" —
this project is exclusively for Cellpose 3 (cyto3) architecture.
