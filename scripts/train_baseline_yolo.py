import os
from pathlib import Path
from dotenv import load_dotenv
import wandb
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

def train_baseline():
    print("Initializing Weights & Biases environment...")

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "road-damage-classification"),
        entity=os.getenv("WANDB_ENTITY"),
        name="yolov8n_baseline_run1",
        job_type="baseline_training"
    )

    print("Initializing YOLOv8n (Nano) model - Baseline...")
    model = YOLO("yolov8n.pt") #dopisac

    results = model.train(
        data=str(PROJECT_ROOT / "road_damage.yaml"),
        epochs=30,
        patience=8,
        imgsz=640,
        batch=16,
        project="yolo_logs",
        name="baseline_run",
        device=0
    )

    wandb.finish()
    print("\nTraining completed successfully! Results saved locally and synced to W&B.")

if __name__ == "__main__":
    train_baseline()