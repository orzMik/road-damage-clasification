import os
from pathlib import Path
from dotenv import load_dotenv
import wandb
from ultralytics import YOLO, settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

def train_yolo26():
    print("Inicjalizacja środowiska Weights & Biases...")

    # Logujemy to jako zupełnie nowy eksperyment w tym samym projekcie
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "road-damage-classification"),
        entity=os.getenv("WANDB_ENTITY"),
        name="yolo26n_comparison_run",
        job_type="train"
    )

    # Zmuszamy YOLO do czytania z głównego folderu
    settings.update({'datasets_dir': str(PROJECT_ROOT)})

    print("Inicjalizacja najnowszego modelu YOLO26n (Nano)...")
    
    # Inicjalizacja architektury YOLO26
    model = YOLO("yolo26n.pt") 

    results = model.train(
        data=str(PROJECT_ROOT / "road_damage_local.yaml"), # Nasz bezpieczny plik
        epochs=30,
        patience=8,
        imgsz=640,
        batch=8, # Zmniejszony batch dla bezpieczeństwa pamięci VRAM
        project="yolo_logs",
        name="yolo26_run",
        device=0
    )

    wandb.finish()
    print("\nTrening YOLO26 zakończony pomyślnie!")

if __name__ == "__main__":
    train_yolo26()