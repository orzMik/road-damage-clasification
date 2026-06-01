"""
Train RF-DETR baseline on the Rome road damage dataset.

This is the RF-DETR equivalent of train_baseline_yolo.py — same dataset,
same splits, different architecture. Designed to be compared head-to-head
with the YOLOv8n baseline.

Requirements:
- CUDA GPU (won't run on MPS or CPU efficiently)
- rfdetr package: pip install "rfdetr[train,loggers]"
- COCO splits prepared: run prepare_coco_split.py first

W&B logging (optional):
- Set env var WANDB_PROJECT to enable W&B logging
- Set env var WANDB_ENTITY to log into a team workspace
- Set env var WANDB_API_KEY (or `wandb login` beforehand) for authentication
- Optional: WANDB_RUN_NAME to name this specific run

Run:  python scripts/train_rfdetr_baseline.py
"""
import os
from pathlib import Path

from rfdetr import RFDETRBase

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "coco_split"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "rfdetr_baseline"

# Config — tuned for Colab T4 (~15GB VRAM)
EPOCHS = 15
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 1e-4


def main():
    if not DATASET_DIR.exists():
        print(f"ERROR: {DATASET_DIR} not found.")
        print("Run prepare_coco_split.py first to create COCO splits.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # W&B config — enabled if WANDB_PROJECT env var is set
    wandb_project = os.environ.get("WANDB_PROJECT")
    use_wandb = wandb_project is not None
    wandb_run_name = os.environ.get("WANDB_RUN_NAME")

    print("=== Training RF-DETR (Base) ===")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Epochs:  {EPOCHS}")
    print(f"Batch:   {BATCH_SIZE} (effective {BATCH_SIZE * GRAD_ACCUM_STEPS} with grad accum)")
    print(f"LR:      {LEARNING_RATE}")
    if use_wandb:
        print(f"W&B:     project={wandb_project}, run={wandb_run_name or '(auto)'}")
    else:
        print("W&B:     disabled (set WANDB_PROJECT to enable)")
    print()

    train_kwargs = dict(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LEARNING_RATE,
        output_dir=str(OUTPUT_DIR),
    )
    if use_wandb:
        train_kwargs["wandb"] = True
        train_kwargs["project"] = wandb_project
        if wandb_run_name:
            train_kwargs["run"] = wandb_run_name

    model = RFDETRBase()
    model.train(**train_kwargs)

    print("\n=== Training complete ===")
    print(f"Best weights saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()