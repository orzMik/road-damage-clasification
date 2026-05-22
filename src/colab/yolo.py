from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import wandb

from src.config.constants import CLASSES


@dataclass(frozen=True)
class DriveConfig:
    mount_point: str = "/content/drive"
    drive_root: str = "MyDrive/road-damage"
    datasets_dirname: str = "datasets"
    runs_dirname: str = "runs"
    models_dirname: str = "models"
    artifacts_dirname: str = "artifacts"


@dataclass(frozen=True)
class DrivePaths:
    root: Path
    datasets: Path
    runs: Path
    models: Path
    artifacts: Path


@dataclass(frozen=True)
class WandbConfig:
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    job_type: str | None = None
    api_key: str | None = None
    config: Mapping[str, Any] | None = None
    tags: Sequence[str] | None = None


@dataclass(frozen=True)
class TrainOutput:
    model: Any
    results: Any
    save_dir: Path


def _load_yolo(weights: str | Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(weights))


def mount_drive(mount_point: str = "/content/drive", force_remount: bool = False) -> Path:
    if importlib.util.find_spec("google.colab") is None:
        raise RuntimeError("Google Colab is required to mount Drive.")

    from google.colab import drive

    drive.mount(mount_point, force_remount=force_remount)
    return Path(mount_point)


def ensure_drive_paths(config: DriveConfig) -> DrivePaths:
    mount_point = Path(config.mount_point)
    if not mount_point.exists():
        raise FileNotFoundError(
            f"Drive mount not found at {config.mount_point}. Run mount_drive(...) first."
        )

    root = mount_point / config.drive_root
    datasets = root / config.datasets_dirname
    runs = root / config.runs_dirname
    models = root / config.models_dirname
    artifacts = root / config.artifacts_dirname

    for path in (root, datasets, runs, models, artifacts):
        path.mkdir(parents=True, exist_ok=True)

    return DrivePaths(
        root=root,
        datasets=datasets,
        runs=runs,
        models=models,
        artifacts=artifacts,
    )


def create_yolo_data_yaml(
    output_path: str | Path,
    dataset_root: str | Path,
    class_names: Sequence[str] | None = None,
) -> Path:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if class_names is None:
        class_names = CLASSES

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"path: {dataset_root.as_posix()}",
        "",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        "names:",
    ]
    lines.extend([f"  {idx}: {name}" for idx, name in enumerate(class_names)])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def setup_wandb(config: WandbConfig) -> wandb.sdk.wandb_run.Run:
    """Authenticate and enable Ultralytics W&B logging for training.

    Ultralytics registers its W&B callbacks at import time, so this must run
    before the first ``YOLO(...)`` call. ``project_dir`` in ``train_yolo`` is
    only the local save path; W&B project/entity/name come from this config.
    """
    if config.api_key:
        wandb.login(key=config.api_key)

    from ultralytics import settings

    settings.update({"wandb": True})

    project = config.project or os.getenv("WANDB_PROJECT")
    if not project:
        raise ValueError(
            "W&B project is required. Set WANDB_PROJECT or pass WandbConfig.project."
        )

    entity = config.entity or os.getenv("WANDB_ENTITY")
    config_payload = dict(config.config) if config.config is not None else None
    tags = list(config.tags) if config.tags is not None else None

    if wandb.run is not None:
        return wandb.run

    return wandb.init(
        project=project,
        entity=entity,
        name=config.name,
        job_type=config.job_type,
        config=config_payload,
        tags=tags,
    )


def init_wandb(config: WandbConfig) -> wandb.sdk.wandb_run.Run:
    """Deprecated alias for :func:`setup_wandb`."""
    return setup_wandb(config)


def train_yolo(
    weights: str | Path,
    data_yaml: str | Path,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int | None = None,
    device: int | str | None = None,
    project_dir: str | Path | None = None,
    run_name: str | None = None,
    wandb_cfg: WandbConfig | None = None,
    **train_kwargs: Any,
) -> TrainOutput:
    if wandb_cfg is not None:
        setup_wandb(wandb_cfg)

    model = _load_yolo(weights)
    args: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
    }
    if batch is not None:
        args["batch"] = batch
    if device is not None:
        args["device"] = device
    if project_dir is not None:
        args["project"] = str(project_dir)
    if run_name is not None:
        args["name"] = run_name
    args.update(train_kwargs)

    results = model.train(**args)
    save_dir_value = getattr(results, "save_dir", None)
    if save_dir_value is None:
        base_dir = Path(args.get("project", "runs"))
        save_dir = base_dir / (args.get("name") or "train")
    else:
        save_dir = Path(save_dir_value)

    return TrainOutput(model=model, results=results, save_dir=save_dir)


def evaluate_yolo(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    device: int | str | None = None,
    project_dir: str | Path | None = None,
    run_name: str | None = None,
    **val_kwargs: Any,
) -> Any:
    model = _load_yolo(weights)
    args: dict[str, Any] = {
        "data": str(data_yaml),
        "split": split,
    }
    if device is not None:
        args["device"] = device
    if project_dir is not None:
        args["project"] = str(project_dir)
    if run_name is not None:
        args["name"] = run_name
    args.update(val_kwargs)

    return model.val(**args)


def predict_yolo(
    weights: str | Path,
    source: str | Path | Iterable[str | Path],
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: int | str | None = None,
    project_dir: str | Path | None = None,
    run_name: str | None = None,
    save: bool = True,
    save_txt: bool = False,
    **predict_kwargs: Any,
) -> Any:
    model = _load_yolo(weights)
    if isinstance(source, (str, Path)):
        source_value: Any = str(source)
    else:
        source_value = [str(item) for item in source]

    args: dict[str, Any] = {
        "source": source_value,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "save": save,
        "save_txt": save_txt,
    }
    if device is not None:
        args["device"] = device
    if project_dir is not None:
        args["project"] = str(project_dir)
    if run_name is not None:
        args["name"] = run_name
    args.update(predict_kwargs)

    return model.predict(**args)
