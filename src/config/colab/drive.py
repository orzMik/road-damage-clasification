from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path


@dataclass(frozen=True)
class DriveConfig:
    mount_point: str = "/content/drive"
    drive_root: str = "MyDrive/road-damage-detection"
    data_dirname: str = "data"
    processed_dirname: str = "processed-yolo"
    runs_dirname: str = "runs"
    models_dirname: str = "models"
    wroclaw_images_dirname: str = "wroclaw_images"


@dataclass(frozen=True)
class DrivePaths:
    root: Path
    data: Path
    processed_yolo: Path
    runs: Path
    models: Path
    wroclaw_images: Path


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
    data = root / config.data_dirname
    processed_yolo = data / config.processed_dirname
    runs = root / config.runs_dirname
    models = root / config.models_dirname
    wroclaw_images = root / config.wroclaw_images_dirname

    for path in (root, data, runs, models, wroclaw_images):
        path.mkdir(parents=True, exist_ok=True)

    return DrivePaths(
        root=root,
        data=data,
        processed_yolo=processed_yolo,
        runs=runs,
        models=models,
        wroclaw_images=wroclaw_images,
    )
