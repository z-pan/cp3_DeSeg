# Cellpose 3 (cyto3) Training & Testing Pipeline

A config-driven pipeline for **Cellpose 3** supporting both:

1. **Segmentation** — fine-tune or evaluate the `cyto3` Res-U-Net model
2. **Image Restoration** — train/run denoise, deblur, or upsample models
   chained with a frozen `cyto3` segmentation backbone

> **Architecture note:** This project targets **Cellpose 3 (`cyto3`)**, not
> Cellpose-SAM (`cpsam`). In cellpose ≥ 4.0 `cpsam` became the default; all
> configs and code explicitly set `pretrained_model="cyto3"`.

---

## Project Structure

```
cp3_DeSeg/
├── configs/
│   ├── default_seg.yaml              # segmentation training/eval
│   ├── default_restore.yaml          # restoration training/eval
│   ├── default_restore_and_seg.yaml  # chained restore → segment pipeline
│   └── experiment/                   # per-experiment overrides (add yours here)
├── src/
│   ├── __init__.py
│   └── config.py                     # YAML loader & dataclass validators
├── scripts/                          # entry-point scripts (train, test, evaluate)
├── notebooks/                        # Jupyter notebooks for visualisation
├── data/                             # symlink or copy your dataset here
│   ├── train/
│   └── test/
├── outputs/
│   ├── models/                       # saved model checkpoints
│   ├── predictions/                  # output masks / restored images
│   └── logs/                         # training logs
├── requirements.txt
└── CLAUDE.md                         # project conventions & architecture rules
```

---

## Setup

### 1. Clone & create environment

```bash
git clone <repo-url> cp3_DeSeg
cd cp3_DeSeg

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
# Install PyTorch first — pick the right CUDA version for your GPU:
# https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install the rest
pip install -r requirements.txt
```

### 3. Verify cellpose installation

```python
import cellpose; print(cellpose.__version__)  # must be >= 3.0, < 5.0

from cellpose import models
model = models.CellposeModel(pretrained_model="cyto3")  # must not raise
```

### 4. Prepare data

Cellpose expects paired files in each data directory:

```
data/train/
    image01.tif        # raw image  (2-D or 3-D)
    image01_seg.npy    # label file (produced by Cellpose GUI or labelme)
    image02.tif
    image02_seg.npy
    ...
```

Symlink an existing dataset:

```bash
ln -s /path/to/your/dataset data/train
```

---

## Configuration

All hyperparameters live in YAML files under `configs/`. Never hardcode them
in scripts.

| File | Purpose |
|---|---|
| `configs/default_seg.yaml` | Segmentation training & eval |
| `configs/default_restore.yaml` | Restoration model training & eval |
| `configs/default_restore_and_seg.yaml` | Chained restore → segment inference |

### Key cyto3 requirements (enforced by `src/config.py`)

| Parameter | Correct value | Why |
|---|---|---|
| `model.pretrained_model` | `"cyto3"` | `"cpsam"` is a different architecture |
| `model.backbone` | `"default"` | `"transformer"` is SegFormer (cpsam only) |
| `eval.channels` | `[cyto_ch, nuc_ch]` | Required — cyto3 is not channel-invariant |
| `eval.diameter` | `> 0` (pixels) | Required — cyto3 is size-dependent |

### Common channel settings

```yaml
eval:
  channels: [0, 0]   # grayscale, no nucleus channel
  channels: [1, 2]   # red = cytoplasm, green = nucleus
  channels: [2, 3]   # green = cytoplasm, blue = nucleus
  channels: [2, 1]   # green = cytoplasm, red = nucleus
```

### Restoration types (`default_restore.yaml`)

```yaml
model:
  restore_type: "denoise_cyto3"   # remove Poisson/Gaussian noise
  restore_type: "deblur_cyto3"    # reverse optical blur
  restore_type: "upsample_cyto3"  # super-resolution upsampling
```

---

## Loading a config in Python

```python
from src.config import load_config

# Auto-detects schema and validates on load
cfg = load_config("configs/default_seg.yaml")

print(cfg.model.pretrained_model)   # "cyto3"
print(cfg.eval.channels)            # [0, 0]
print(cfg.eval.diameter)            # 30.0
print(cfg.training.learning_rate)   # 1e-05

# Restoration config
rcfg = load_config("configs/default_restore.yaml")
print(rcfg.model.restore_type)      # "denoise_cyto3"
print(rcfg.augmentation.poisson)    # 0.8

# Skip validation (useful for partial / draft configs)
cfg = load_config("configs/default_seg.yaml", validate=False)
```

---

## Cellpose 3 API Quick Reference

### Segmentation

```python
from cellpose import models, train

# Load model
model = models.CellposeModel(pretrained_model="cyto3")

# Inference
masks, flows, styles, diams = model.eval(
    images,
    channels=[1, 2],       # REQUIRED
    diameter=30.0,         # REQUIRED
    flow_threshold=0.4,
    cellprob_threshold=0.0,
)

# Fine-tune
model_path, train_losses, test_losses = train.train_seg(
    model.net,
    train_data=images,
    train_labels=labels,
    learning_rate=1e-5,
    weight_decay=0.1,
    n_epochs=100,
)
```

### Restoration (standalone)

```python
from cellpose import denoise

restore_model = denoise.DenoiseModel(
    model_type="cyto3",
    restore_type="denoise_cyto3",
)

restored_imgs = restore_model.eval(images, channels=[1, 2], diameter=30.0)
```

### Restore + Segment (integrated)

```python
from cellpose import denoise

combined = denoise.CellposeDenoiseModel(
    model_type="cyto3",
    restore_type="denoise_cyto3",
)

masks, flows, styles, diams, restored_imgs = combined.eval(
    images, channels=[1, 2], diameter=30.0
)
```

---

## What NOT to do

```python
# WRONG — cpsam is a different architecture (SegFormer/SAM)
models.CellposeModel(pretrained_model="cpsam")

# WRONG — Cellpose class was removed in cellpose 4
models.Cellpose(model_type="cyto3")

# WRONG — channels is required for cyto3
model.eval(images, diameter=30.0)

# WRONG — diameter is required for cyto3
model.eval(images, channels=[1, 2])
```
