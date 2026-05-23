from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.config.wandb import WandbConfig, setup_wandb

CLASSES = ("Pothole", "Crack", "Manhole")


@dataclass(frozen=True)
class TrainOutput:
    model: Any
    results: Any
    save_dir: Path


def _load_yolo(weights: str | Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(weights))


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
