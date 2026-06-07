import os
from pathlib import Path
from dotenv import load_dotenv
import wandb
from ultralytics import YOLO, settings # <-- Zmiana: importujemy settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

def train_yolo11():
    print("Inicjalizacja środowiska Weights & Biases...")

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "road-damage-classification"),
        entity=os.getenv("WANDB_ENTITY"),
        name="yolo11n_comparison_run",
        job_type="train"
    )

    # Zmiana: Zmuszamy YOLO, aby traktowało główny folder projektu jako katalog ze zbiorami danych
    settings.update({'datasets_dir': str(PROJECT_ROOT)})

    print("Inicjalizacja modelu YOLO11n (Nano)...")
    model = YOLO("yolo11n.pt") 

    results = model.train(
        data=str(PROJECT_ROOT / "road_damage_local.yaml"),
        epochs=30,
        patience=8,
        imgsz=640,
        batch=8,
        project="yolo_logs",
        name="yolo11_run",
        device=0
    )

    wandb.finish()
    print("\nTrening YOLO11 zakończony pomyślnie!")

if __name__ == "__main__":
    train_yolo11()