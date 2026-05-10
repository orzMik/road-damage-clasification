from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_IMAGES_DIR = DATA_DIR / "images"
RAW_YOLO_LABELS_DIR = DATA_DIR / "labels-YOLO"

PROCESSED_YOLO_DATA_DIR = DATA_DIR / "processed-yolo"

CLASSES = ["Pothole", "Crack", "Manhole"]
NUM_CLASSES = len(CLASSES)
