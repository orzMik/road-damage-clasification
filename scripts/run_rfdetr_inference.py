"""Run RF-DETR inference on a folder of images (e.g., Wroclaw photos).

Requires CUDA GPU; intended to run on Colab. The trained checkpoint
(checkpoint_best_ema.pth) must be at runs/rfdetr_baseline/.

Run:  python scripts/run_rfdetr_inference.py
"""
import sys
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT = PROJECT_ROOT / "runs" / "rfdetr_baseline" / "checkpoint_best_ema.pth"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "wroclaw"
OUTPUT_DIR = PROJECT_ROOT / "inference_results" / "rfdetr_wroclaw"
CONF_THRESHOLD = 0.25

# Category IDs after remap (see prepare_coco_split.py): 1=pothole, 2=crack, 3=manhole
CLASS_NAMES = {1: "pothole", 2: "crack", 3: "manhole"}


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        image = Image.open(img_path).convert("RGB")
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

    print("=== Detections summary ===")
    print("\n".join(summary))
    print(f"\nAnnotated images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()