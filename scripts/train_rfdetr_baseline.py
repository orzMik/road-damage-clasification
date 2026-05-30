"""
Train RF-DETR baseline on the Rome road damage dataset.

This is the RF-DETR equivalent of train_baseline_yolo.py — same dataset,
same splits, different architecture. Designed to be compared head-to-head
with the YOLOv8n baseline.

Requirements:
- CUDA GPU (won't run on MPS or CPU efficiently)
- rfdetr package: pip install rfdetr
- COCO splits prepared: run prepare_coco_split.py first

Run:  python scripts/train_rfdetr_baseline.py

Known limitations (same as YOLO baseline):
- Random split across video frames creates data leakage between splits.
- Pipeline smoke test; meant for comparison with YOLOv8n baseline.
"""
from pathlib import Path

from rfdetr import RFDETRBase

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "coco_split"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "rfdetr_baseline"

# Config — tuned for Colab T4 (~15GB VRAM)
EPOCHS = 15
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 1e-4


def main():
    if not DATASET_DIR.exists():
        print(f"ERROR: {DATASET_DIR} not found.")
        print("Run prepare_coco_split.py first to create COCO splits.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Training RF-DETR (Base) ===")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Epochs:  {EPOCHS}")
    print(f"Batch:   {BATCH_SIZE} (effective {BATCH_SIZE * GRAD_ACCUM_STEPS} with grad accum)")
    print(f"LR:      {LEARNING_RATE}")
    print()

    model = RFDETRBase()
    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LEARNING_RATE,
        output_dir=str(OUTPUT_DIR),
    )

    print("\n=== Training complete ===")
    print(f"Best weights saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()