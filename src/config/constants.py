from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_IMAGES_DIR = BASE_DIR / "data" / "raw" / "images"
RAW_YOLO_LABELS_DIR = BASE_DIR / "data" / "raw" / "labels"
PROCESSED_YOLO_DATA_DIR = BASE_DIR / "data" / "processed" / "yolo"

CLASSES = ["Pothole", "Manhole", "Crack"]