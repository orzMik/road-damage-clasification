"""Run RF-DETR inference locally on Mac (CPU mode) with EXIF orientation fix.

Use for small batches only — CPU inference is slow for transformer models.
"""
import sys
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageOps
from rfdetr import RFDETRBase

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT = PROJECT_ROOT / "runs" / "rfdetr_baseline" / "checkpoint_best_ema.pth"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "wroclaw_mikolaj"
OUTPUT_DIR = PROJECT_ROOT / "inference_results" / "rfdetr_wroclaw_mikolaj"
CONF_THRESHOLD = 0.25

CLASS_NAMES = {0: "pothole", 1: "crack", 2: "manhole"}


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch device: {'cuda' if torch.cuda.is_available() else 'cpu (slow but works)'}")
    print(f"Loading model from {CHECKPOINT}")
    model = RFDETRBase(pretrain_weights=str(CHECKPOINT), num_classes=3)

    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    print(f"Running inference on {len(image_paths)} images from {input_dir}\n")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    summary = []
    for img_path in image_paths:
        print(f"  Processing {img_path.name}...")
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)  # Apply EXIF rotation
        detections = model.predict(image, threshold=CONF_THRESHOLD)
        labels = [
            f"{CLASS_NAMES.get(int(c), '?')}({conf:.2f})"
            for c, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated = np.array(image)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )
        Image.fromarray(annotated).save(OUTPUT_DIR / img_path.name)
        summary.append(
            f"  {img_path.name}: {', '.join(labels) if labels else 'no detections'}"
        )

    print("\n=== Detections summary ===")
    print("\n".join(summary))
    print(f"\nAnnotated images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()