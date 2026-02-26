"""Config loader and validator for the Cellpose 3 pipeline.

Supports three config schemas:
    - SegConfig           (default_seg.yaml)
    - RestoreConfig       (default_restore.yaml)
    - RestoreAndSegConfig (default_restore_and_seg.yaml)

Usage:
    from src.config import load_config, SegConfig, RestoreConfig, RestoreAndSegConfig

    cfg = load_config("configs/default_seg.yaml")
    # cfg is a SegConfig dataclass instance

    # Access fields:
    lr = cfg.training.learning_rate
    ch = cfg.eval.channels
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PRETRAINED_SEG = {"cyto3"}           # cpsam deliberately excluded
VALID_BACKBONES = {"default", "transformer"}
VALID_RESTORE_TYPES = {
    "denoise_cyto3",
    "deblur_cyto3",
    "upsample_cyto3",
    "denoise_nuclei",
    "deblur_nuclei",
    "upsample_nuclei",
}
VALID_NOISE_TYPES = {"poisson", "gaussian", "mixed"}

# ---------------------------------------------------------------------------
# Shared sub-configs
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    channels: List[int] = field(default_factory=lambda: [0, 0])
    diameter: float = 30.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    do_3D: bool = False
    niter: Optional[int] = None

    def validate(self) -> None:
        if len(self.channels) != 2:
            raise ValueError(
                f"eval.channels must be a list of exactly 2 integers, got {self.channels}"
            )
        if not all(isinstance(c, int) and c >= 0 for c in self.channels):
            raise ValueError(
                f"eval.channels values must be non-negative integers, got {self.channels}"
            )
        if self.diameter <= 0:
            raise ValueError(f"eval.diameter must be > 0, got {self.diameter}")


@dataclass
class SegTrainingConfig:
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.1
    n_epochs: int = 100
    batch_size: int = 8
    min_train_masks: int = 5
    save_every: int = 50
    save_path: str = "outputs/models"
    model_name: str = "cyto3_seg_custom"

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"training.learning_rate must be > 0, got {self.learning_rate}")
        if self.n_epochs <= 0:
            raise ValueError(f"training.n_epochs must be > 0, got {self.n_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"training.batch_size must be > 0, got {self.batch_size}")


@dataclass
class DataConfig:
    train_dir: str = "data/train"
    test_dir: Optional[str] = "data/test"
    image_filter: str = ".tif"
    mask_filter: str = "_seg.npy"


# ---------------------------------------------------------------------------
# SegConfig
# ---------------------------------------------------------------------------

@dataclass
class SegModelConfig:
    pretrained_model: str = "cyto3"
    backbone: str = "default"

    def validate(self) -> None:
        if self.pretrained_model not in VALID_PRETRAINED_SEG:
            raise ValueError(
                f"model.pretrained_model must be one of {VALID_PRETRAINED_SEG}, "
                f"got '{self.pretrained_model}'. "
                "This pipeline is exclusively for Cellpose 3 (cyto3). "
                "Do NOT use 'cpsam'."
            )
        if self.backbone not in VALID_BACKBONES:
            raise ValueError(
                f"model.backbone must be one of {VALID_BACKBONES}, got '{self.backbone}'"
            )


@dataclass
class SegConfig:
    """Full configuration for segmentation training / inference."""

    model: SegModelConfig = field(default_factory=SegModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    training: SegTrainingConfig = field(default_factory=SegTrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> None:
        self.model.validate()
        self.eval.validate()
        self.training.validate()
        logger.info(
            "SegConfig validated: pretrained_model=%s, channels=%s, diameter=%s",
            self.model.pretrained_model,
            self.eval.channels,
            self.eval.diameter,
        )


# ---------------------------------------------------------------------------
# RestoreConfig
# ---------------------------------------------------------------------------

@dataclass
class RestoreModelConfig:
    model_type: str = "cyto3"
    restore_type: str = "denoise_cyto3"
    chan2_restore: bool = False

    def validate(self) -> None:
        if self.model_type != "cyto3":
            raise ValueError(
                f"model.model_type must be 'cyto3', got '{self.model_type}'. "
                "Restoration is not supported for cpsam."
            )
        if self.restore_type not in VALID_RESTORE_TYPES:
            raise ValueError(
                f"model.restore_type must be one of {VALID_RESTORE_TYPES}, "
                f"got '{self.restore_type}'"
            )


@dataclass
class SegModelRefConfig:
    """Reference to the frozen segmentation model used during restore training."""
    model_type: str = "cyto3"

    def validate(self) -> None:
        if self.model_type != "cyto3":
            raise ValueError(
                f"seg_model.model_type must be 'cyto3', got '{self.model_type}'"
            )


@dataclass
class AugmentationConfig:
    noise_type: str = "poisson"
    poisson: float = 0.8
    blur: float = 0.0
    downsample: int = 1

    def validate(self) -> None:
        if self.noise_type not in VALID_NOISE_TYPES:
            raise ValueError(
                f"augmentation.noise_type must be one of {VALID_NOISE_TYPES}, "
                f"got '{self.noise_type}'"
            )
        if not 0.0 <= self.poisson <= 1.0:
            raise ValueError(
                f"augmentation.poisson must be in [0, 1], got {self.poisson}"
            )
        if self.blur < 0:
            raise ValueError(f"augmentation.blur must be >= 0, got {self.blur}")
        if self.downsample < 1:
            raise ValueError(
                f"augmentation.downsample must be >= 1, got {self.downsample}"
            )


@dataclass
class RestoreTrainingConfig:
    learning_rate: float = 0.001
    weight_decay: float = 1.0e-5
    n_epochs: int = 2000
    batch_size: int = 8
    save_every: int = 200
    save_path: str = "outputs/models"
    model_name: str = "denoise_cyto3_custom"

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"training.learning_rate must be > 0, got {self.learning_rate}")
        if self.n_epochs <= 0:
            raise ValueError(f"training.n_epochs must be > 0, got {self.n_epochs}")


@dataclass
class RestoreConfig:
    """Full configuration for restoration model training / inference."""

    model: RestoreModelConfig = field(default_factory=RestoreModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seg_model: SegModelRefConfig = field(default_factory=SegModelRefConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: RestoreTrainingConfig = field(default_factory=RestoreTrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> None:
        self.model.validate()
        self.eval.validate()
        self.seg_model.validate()
        self.augmentation.validate()
        self.training.validate()
        logger.info(
            "RestoreConfig validated: model_type=%s, restore_type=%s, channels=%s",
            self.model.model_type,
            self.model.restore_type,
            self.eval.channels,
        )


# ---------------------------------------------------------------------------
# RestoreAndSegConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    name: str = "restore_then_seg"
    use_integrated_model: bool = True


@dataclass
class RestoreStageConfig:
    model_type: str = "cyto3"
    restore_type: str = "denoise_cyto3"
    chan2_restore: bool = False
    checkpoint: Optional[str] = None

    def validate(self) -> None:
        if self.model_type != "cyto3":
            raise ValueError(
                f"restore.model_type must be 'cyto3', got '{self.model_type}'"
            )
        if self.restore_type not in VALID_RESTORE_TYPES:
            raise ValueError(
                f"restore.restore_type must be one of {VALID_RESTORE_TYPES}, "
                f"got '{self.restore_type}'"
            )


@dataclass
class SegStageConfig:
    pretrained_model: str = "cyto3"
    backbone: str = "default"
    checkpoint: Optional[str] = None

    def validate(self) -> None:
        if self.pretrained_model not in VALID_PRETRAINED_SEG:
            raise ValueError(
                f"seg.pretrained_model must be one of {VALID_PRETRAINED_SEG}, "
                f"got '{self.pretrained_model}'"
            )


@dataclass
class InputDataConfig:
    input_dir: str = "data/test"
    image_filter: str = ".tif"


@dataclass
class OutputConfig:
    save_path: str = "outputs/predictions"
    save_masks: bool = True
    save_restored: bool = True
    save_flows: bool = False


@dataclass
class RestoreAndSegConfig:
    """Full configuration for the chained restore → segment pipeline."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    restore: RestoreStageConfig = field(default_factory=RestoreStageConfig)
    seg: SegStageConfig = field(default_factory=SegStageConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: InputDataConfig = field(default_factory=InputDataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.restore.validate()
        self.seg.validate()
        self.eval.validate()
        logger.info(
            "RestoreAndSegConfig validated: restore_type=%s, seg_model=%s, "
            "integrated=%s, channels=%s",
            self.restore.restore_type,
            self.seg.pretrained_model,
            self.pipeline.use_integrated_model,
            self.eval.channels,
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

ConfigType = Union[SegConfig, RestoreConfig, RestoreAndSegConfig]

# Keys present in each YAML that uniquely identify its schema.
_SCHEMA_DISCRIMINATORS: dict[str, type] = {
    "seg": SegConfig,          # has top-level "model.pretrained_model"
    "restore": RestoreConfig,  # has top-level "model.restore_type"
    "restore_and_seg": RestoreAndSegConfig,  # has top-level "pipeline"
}


def _detect_schema(raw: dict) -> type:
    """Infer the config schema from the YAML structure."""
    if "pipeline" in raw:
        return RestoreAndSegConfig
    model_block = raw.get("model", {})
    if "restore_type" in model_block:
        return RestoreConfig
    if "pretrained_model" in model_block:
        return SegConfig
    raise ValueError(
        "Cannot detect config schema. YAML must contain one of: "
        "'pipeline' (restore_and_seg), 'model.restore_type' (restore), "
        "or 'model.pretrained_model' (seg)."
    )


def _populate_dataclass(cls: type, data: dict):
    """Recursively build a dataclass from a nested dict."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data  # leaf value

    field_map = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for name, fld in field_map.items():
        if name not in data:
            # Use the dataclass default
            if fld.default is not dataclasses.MISSING:
                kwargs[name] = fld.default
            elif fld.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                kwargs[name] = fld.default_factory()
            # else: required field missing — let the dataclass raise
        else:
            val = data[name]
            # Resolve the actual field type (unwrap Optional, etc.)
            ftype = fld.type
            if isinstance(ftype, str):
                # Forward reference — resolve against globals
                import sys
                ftype = eval(ftype, sys.modules[__name__].__dict__)  # noqa: S307
            origin = getattr(ftype, "__origin__", None)
            # Unwrap Optional[X] → X
            if origin is Union:
                args = [a for a in ftype.__args__ if a is not type(None)]
                ftype = args[0] if args else ftype
                origin = getattr(ftype, "__origin__", None)
            if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
                kwargs[name] = _populate_dataclass(ftype, val)
            else:
                kwargs[name] = val
    return cls(**kwargs)


def load_config(path: Union[str, Path], validate: bool = True) -> ConfigType:
    """Load a YAML config file and return a validated dataclass instance.

    Args:
        path: Path to the YAML config file.
        validate: If True (default), call cfg.validate() after loading.

    Returns:
        A SegConfig, RestoreConfig, or RestoreAndSegConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config fails schema detection or validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.info("Loading config from %s", path)
    with path.open("r") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    schema_cls = _detect_schema(raw)
    cfg = _populate_dataclass(schema_cls, raw)

    if validate:
        cfg.validate()

    return cfg
