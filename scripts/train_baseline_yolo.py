"""
Baseline YOLOv8 for road damage detection (pothole/crack/manhole).
Trains on Rome dataset with random train/val/test split.

Run:  uv run scripts/train_baseline_yolo.py

Known limitations:
- Random split across video frames creates data leakage between splits
  (consecutive frames are nearly identical). Use group-aware split for
  a proper experiment. This is a pipeline smoke test only.
"""

import random
import shutil
from pathlib import Path

from ultralytics import YOLO

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels-YOLO"
SPLIT_DIR = DATA_DIR / "yolo_split"

# Config
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42
EPOCHS = 30
IMG_SIZE = 640
BATCH = 16
DEVICE = "mps"  # Apple Silicon. Use "cuda" on Linux/Windows GPU, "cpu" otherwise.
MODEL = "yolov8n.pt"  # nano = fastest. Use "yolov8s.pt" for slightly better mAP.

CLASS_NAMES = ["pothole", "crack", "manhole"]


def create_splits() -> None:
    """Create train/val/test split using symlinks. Idempotent."""
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    for split in ("train", "val", "test"):
        (SPLIT_DIR / "images" / split).mkdir(parents=True)
        (SPLIT_DIR / "labels" / split).mkdir(parents=True)

    # Only keep images that actually have a label file
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    paired = [img for img in images if (LABELS_DIR / f"{img.stem}.txt").exists()]
    print(f"Found {len(images)} images, {len(paired)} with YOLO labels")

    random.seed(SEED)
    random.shuffle(paired)

    n = len(paired)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    splits = {
        "train": paired[:n_train],
        "val": paired[n_train : n_train + n_val],
        "test": paired[n_train + n_val :],
    }

    for split_name, files in splits.items():
        for img_path in files:
            label_path = LABELS_DIR / f"{img_path.stem}.txt"
            (SPLIT_DIR / "images" / split_name / img_path.name).symlink_to(img_path.resolve())
            (SPLIT_DIR / "labels" / split_name / label_path.name).symlink_to(label_path.resolve())
        print(f"  {split_name}: {len(files)} images")


def write_data_yaml() -> Path:
    """Write data.yaml that ultralytics needs."""
    yaml_path = SPLIT_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {SPLIT_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"Wrote {yaml_path}")
    return yaml_path


def main() -> None:
    print("=== 1. Creating train/val/test splits ===")
    create_splits()

    print("\n=== 2. Writing data.yaml ===")
    yaml_path = write_data_yaml()

    print(f"\n=== 3. Training YOLOv8 ({MODEL}) on {DEVICE} ===")
    model = YOLO(MODEL)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project="runs/baseline",
        name="yolov8n_rome",
        exist_ok=True,
    )

    print("\n=== 4. Evaluating on test set ===")
    metrics = model.val(data=str(yaml_path), split="test")
    print(f"\nTest mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"Test mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"\nPer-class mAP@0.5:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:10s}: {metrics.box.maps[i]:.4f}")


if __name__ == "__main__":
    main()