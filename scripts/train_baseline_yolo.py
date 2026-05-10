import os
from pathlib import Path
from dotenv import load_dotenv
import wandb
from ultralytics import YOLO

# Ensure the script correctly identifies the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from the .env file (hidden keys and project names)
load_dotenv(PROJECT_ROOT / ".env")

def train_baseline():
    print("Initializing Weights & Biases environment...")
    
    # Manual W&B initialization to ensure logs go to the Team Workspace
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "road-damage-classification"),
        entity=os.getenv("WANDB_ENTITY"), # Links the run to the team
        name="yolov8n_baseline_run1",     # Name of this specific training run
        job_type="baseline_training"
    )

    print("Initializing YOLOv8n (Nano) model - Baseline...")
    model = YOLO("yolov8n.pt")
    
    # YOLO will automatically detect the active wandb session and attach to it
    results = model.train(
        data=str(PROJECT_ROOT / "road_damage.yaml"),
        epochs=30,
        patience=8,               # Early stopping if no improvement for 8 epochs
        imgsz=640,
        batch=16,
        project="yolo_logs",      # Local folder for YOLO logs
        name="baseline_run",      # Local subfolder name for this run
        device=0                  # Device ID for GPU (use 'cpu' if no GPU is available)
    )
    
    # Gracefully close the synchronization session after training
    wandb.finish()
    print("\nTraining completed successfully! Results saved locally and synced to W&B.")

if __name__ == "__main__":
    train_baseline()