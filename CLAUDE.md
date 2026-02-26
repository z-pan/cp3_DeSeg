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

---

## 修正后的分阶段 Prompt

### 阶段 0：让 Claude Code 理解正确的 API
```
Read the following cellpose documentation pages and summarize the key API
differences between Cellpose 3 (cyto3) and Cellpose-SAM (cpsam):

1. https://cellpose.readthedocs.io/en/latest/train.html (training API)
2. https://cellpose.readthedocs.io/en/latest/restore.html (restoration API)
3. https://cellpose.readthedocs.io/en/latest/api.html (full API reference)
4. https://cellpose.readthedocs.io/en/latest/settings.html (settings & model differences)

Key focus:
- We are building a pipeline for Cellpose 3 (cyto3), NOT Cellpose-SAM
- We need BOTH segmentation AND image restoration (denoise/deblur/upsample)
- Summarize: CellposeModel with cyto3, CellposeDenoiseModel, DenoiseModel,
  train.train_seg(), denoise.DenoiseModel.train()
- Note the correct default hyperparameters for cyto3 training
```

### 阶段 1：项目脚手架（修正版）
```
Initialize the project structure for a Cellpose 3 (cyto3) training and
testing pipeline that supports BOTH segmentation AND image restoration.

Create:
1. Project directory structure as defined in CLAUDE.md
2. `configs/default_seg.yaml` — segmentation training config:
   - model: pretrained_model: "cyto3" (NOT cpsam)
   - channels: [0, 0]  # grayscale; user can change to [1,2] for cyto+nuclei
   - diameter: 30
   - training: learning_rate: 1e-5, weight_decay: 0.1, n_epochs: 100
3. `configs/default_restore.yaml` — restoration training config:
   - model_type: "cyto3"
   - restore_type: "denoise_cyto3"  # options: denoise_cyto3, deblur_cyto3, upsample_cyto3
   - chan2_restore: false
   - noise_type: "poisson"  # training noise augmentation
   - seg_model_type: "cyto3"  # frozen segmentation model for loss
   - training: learning_rate: 0.001, n_epochs: 2000
4. `configs/default_restore_and_seg.yaml` — chained pipeline config
5. `src/config.py` — loads and validates YAML configs
6. `requirements.txt` — pin cellpose>=3.0,<5.0
7. `README.md` with setup instructions

Do NOT implement training/testing logic yet — just the scaffolding.
```

### 阶段 2：数据加载（同前，无变化）

与之前的阶段 2 prompt 相同——数据加载逻辑不受架构差异影响。

### 阶段 3：Segmentation 训练模块（修正版）
```
Create `src/seg_trainer.py` and `scripts/train_seg.py` —
the segmentation training module for Cellpose 3 (cyto3).

⚠️ CRITICAL: Use pretrained_model="cyto3", NOT "cpsam".

Requirements for `src/seg_trainer.py`:
1. A `CellposeSegTrainer` class:
   - __init__(self, cfg): initializes CellposeModel with
     pretrained_model="cyto3" (explicitly, not default)
     Sets gpu, channels, diameter from config
   - train(self): runs train.train_seg with cyto3 model
   - Key cyto3-specific settings:
     * channels must be explicitly provided (e.g., [0,0] or [1,2])
     * diameter must be set (cyto3 is NOT size-invariant)
     * normalize settings may need tile_norm_blocksize for crops

2. Training features:
   - Support fine-tuning from cyto3, cyto2, nuclei, or custom model path
   - Do NOT allow fine-tuning from cpsam (different architecture)
   - Save best model based on test loss
   - Save loss curves, config snapshot, training metadata

3. API usage (correct for cyto3):
```python
   from cellpose import models, train
   model = models.CellposeModel(gpu=True, pretrained_model="cyto3")
   model_path, train_losses, test_losses = train.train_seg(
       model.net,
       train_data=images, train_labels=labels,
       test_data=test_images, test_labels=test_labels,
       channels=cfg.channels,
       weight_decay=cfg.weight_decay,
       learning_rate=cfg.learning_rate,
       n_epochs=cfg.n_epochs,
       model_name=cfg.model_name
   )
```

Requirements for `scripts/train_seg.py`:
1. CLI entry point: --config, optional CLI overrides
2. Validates that pretrained_model is cyto3-compatible (not cpsam)
3. Logs all settings, runs training, saves outputs
```

### 阶段 4：Restoration 训练模块（新增）
```
Create `src/restore_trainer.py` and `scripts/train_restore.py` —
the image restoration training module. This is a Cellpose 3-specific feature.

⚠️ This module implements the Cellpose 3 restoration architecture:
   trainable restoration network → frozen segmentation network,
   trained with segmentation loss + perceptual loss.

Requirements for `src/restore_trainer.py`:
1. A `CellposeRestoreTrainer` class:
   - __init__(self, cfg): initializes denoise.DenoiseModel with
     model_type=cfg.model_type (default "cyto3")
   - train(self): runs DenoiseModel.train() with config parameters
   - Support three restore types: denoise, deblur, upsample
   - Support training noise augmentation parameters:
     poisson noise level, gaussian blur, downsample factor

2. API usage (correct for Cellpose 3 restoration):
```python
   from cellpose import denoise
   model = denoise.DenoiseModel(gpu=True, nchan=1)
   model_path = model.train(
       train_data, train_labels,
       test_data=test_data, test_labels=test_labels,
       save_path=save_path,
       iso=True,
       blur=0.,       # set >0 for deblur training
       downsample=0.,  # set >0 for upsample training
       poisson=0.8,    # Poisson noise level
       n_epochs=2000,
       learning_rate=0.001,
       seg_model_type="cyto3"  # frozen seg model for loss computation
   )
```

3. For combined denoise + segmentation inference:
```python
   from cellpose import denoise
   model = denoise.CellposeDenoiseModel(
       gpu=True,
       model_type="cyto3",
       restore_type="denoise_cyto3",
       chan2_restore=False
   )
   masks, flows, styles, imgs_restored = model.eval(
       imgs, channels=[0,0], diameter=30.
   )
```

Requirements for `scripts/train_restore.py`:
1. CLI: --config, --restore_type (denoise/deblur/upsample)
2. Validates model_type is cyto3, not cpsam
3. Logs training, saves model and restored image examples
```

### 阶段 5：测试与评估（修正版）
```
Create `src/evaluator.py` and `scripts/test.py` —
evaluation for BOTH segmentation and restoration.

Requirements for `src/evaluator.py`:
1. `SegEvaluator` class — evaluates segmentation quality:
   - Loads CellposeModel with pretrained_model="cyto3" or custom model path
   - Runs model.eval() with channels and diameter from config
   - Metrics: AP@[0.5, 0.75, 0.9] via cellpose.metrics.average_precision,
     AJI via cellpose.metrics.aggregated_jaccard_index,
     boundary F1 via cellpose.metrics.boundary_scores,
     flow error, cell count comparison
   - Visualization: overlays, side-by-side, error maps

2. `RestoreEvaluator` class — evaluates restoration quality:
   - Loads CellposeDenoiseModel with restore_type from config
   - Runs model.eval() which returns both masks AND restored images
   - Image quality metrics: PSNR, SSIM (restored vs clean if available)
   - Segmentation improvement metrics: AP before restoration vs after
   - Visualization: original → restored → segmented pipeline view

3. `RestoreAndSegEvaluator` class — evaluates the full pipeline:
   - Step 1: Restore images using DenoiseModel or CellposeDenoiseModel
   - Step 2: Segment restored images using CellposeModel(pretrained_model="cyto3")
   - Compare: segmentation on raw images vs segmentation on restored images
   - Report the delta in AP, AJI, boundary F1

4. Results export: per-image CSV, summary JSON, _seg.npy files,
   all visualizations to output directory

Requirements for `scripts/test.py`:
1. CLI: --config, --model_path, --test_dir, --mode (seg/restore/full)
2. Mode selection determines which evaluator to use
```

### 阶段 6：实验管理（同前，小修正）

与之前的阶段 5 prompt 相同，但新增 restoration 实验的对比：
```
Additionally, create experiment configs:
- `configs/experiment/finetune_cyto3_seg.yaml` — segmentation fine-tuning
- `configs/experiment/train_denoise.yaml` — denoise restoration training
- `configs/experiment/train_deblur.yaml` — deblur restoration training
- `configs/experiment/train_upsample.yaml` — upsample restoration training
- `configs/experiment/restore_then_seg.yaml` — chained pipeline

Do NOT create any config with pretrained_model="cpsam" —
this project is exclusively for Cellpose 3 (cyto3) architecture.
