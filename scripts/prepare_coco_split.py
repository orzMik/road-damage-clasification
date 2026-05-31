"""Prepare COCO splits for RF-DETR training.

Splits data/raw/annotations_coco.json into train/valid/test using the SAME seed
as the YOLO baseline (42, 70/15/15) so results are directly comparable.

Remaps category IDs from {0,1,2} to {1,2,3} because RF-DETR (following Roboflow
COCO convention) treats id=0 as background.

Run:  uv run scripts/prepare_coco_split.py
"""
import json
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
COCO_JSON = RAW_DIR / "annotations_coco.json"
OUTPUT_DIR = DATA_DIR / "coco_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42


def main():
    if not COCO_JSON.exists():
        print(f"ERROR: {COCO_JSON} not found. Did you run download_dataset.py?")
        return

    print(f"Loading {COCO_JSON}")
    with open(COCO_JSON) as f:
        coco = json.load(f)

    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    # Remap category IDs: original {0,1,2} -> new {1,2,3}
    # RF-DETR treats category_id=0 as background by convention
    id_remap = {cat["id"]: cat["id"] + 1 for cat in coco["categories"]}
    new_categories = [
        {**cat, "id": id_remap[cat["id"]]} for cat in coco["categories"]
    ]
    new_annotations = [
        {**ann, "category_id": id_remap[ann["category_id"]]}
        for ann in coco["annotations"]
    ]
    print(f"  Categories after remap: {[(c['id'], c['name']) for c in new_categories]}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    images = coco["images"]
    random.seed(SEED)
    shuffled = images.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    splits = {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    for split_name, split_images in splits.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True)
        image_ids = {img["id"] for img in split_images}
        split_annotations = [a for a in new_annotations if a["image_id"] in image_ids]

        for img in split_images:
            src = IMAGES_DIR / img["file_name"]
            dst = split_dir / img["file_name"]
            if not src.exists():
                print(f"  WARNING: source image not found: {src}")
                continue
            dst.symlink_to(src.resolve())

        split_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": new_categories,
            "images": split_images,
            "annotations": split_annotations,
        }
        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(split_coco, f)

        print(f"  {split_name}: {len(split_images)} images, {len(split_annotations)} annotations")

    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()