import sys
import shutil
import random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config.constants import RAW_IMAGES_DIR, RAW_YOLO_LABELS_DIR, PROCESSED_YOLO_DATA_DIR, CLASSES

def count_class_instances(pairs):
    """Counts instances of each class in the given list of image-label pairs."""
    counts = {i: 0 for i in range(len(CLASSES))}
    
    for _, label_path in pairs:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id in counts:
                        counts[class_id] += 1
    return counts

def split_dataset(
    train_ratio: float = 0.7, 
    val_ratio: float = 0.2
):
    """
    Splits the dataset into train/val/test and copies it into a new structure.
    """
    if not RAW_IMAGES_DIR.exists() or not RAW_YOLO_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_IMAGES_DIR} or {RAW_YOLO_LABELS_DIR}. "
            "Make sure the dataset is extracted correctly into the 'data/raw' folder."
        )

    splits = ["train", "val", "test"]
    for split in splits:
        (PROCESSED_YOLO_DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_YOLO_DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    all_images = list(RAW_IMAGES_DIR.glob("*.jpg"))

    valid_pairs = []
    for img_path in all_images:
        label_path = RAW_YOLO_LABELS_DIR / f"{img_path.stem}.txt"
        
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            print(f"Warning: Missing label for image {img_path.name}. Skipping.")

    random.seed(42)
    random.shuffle(valid_pairs)

    total = len(valid_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    datasets = {
        "train": valid_pairs[:train_end],
        "val": valid_pairs[train_end:val_end],
        "test": valid_pairs[val_end:]
    }

    print("Starting file copy process and calculating stats...\n")
    for split_name, pairs in datasets.items():
        split_images_dir = PROCESSED_YOLO_DATA_DIR / split_name / "images"
        split_labels_dir = PROCESSED_YOLO_DATA_DIR / split_name / "labels"
        
        for img_path, label_path in pairs:
            shutil.copy2(img_path, split_images_dir / img_path.name)
            shutil.copy2(label_path, split_labels_dir / label_path.name)

        class_counts = count_class_instances(pairs)
        total_objects = sum(class_counts.values())
        
        print(f"[{split_name.upper()} SPLIT] - {len(pairs)} images copied.")
        if total_objects > 0:
            for class_id, count in class_counts.items():
                class_name = CLASSES[class_id]
                percentage = (count / total_objects) * 100
                print(f"  -> {class_name:<8}: {count} objects ({percentage:.1f}%)")
        else:
            print("  -> No objects found in this split.")
        print("-" * 40)

    print(f"\nSuccess! Data saved to: {PROCESSED_YOLO_DATA_DIR.absolute()}")


if __name__ == "__main__":
    split_dataset(train_ratio=0.7, val_ratio=0.2)