"""
Inference avec le modèle baseline entraîné.

Run:
    uv run scripts/run_inference.py                 # défaut: data/wroclaw/
    uv run scripts/run_inference.py path/to/images  # custom
"""
import sys
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "wroclaw"
OUTPUT_DIR = PROJECT_ROOT / "inference_results"

CONF_THRESHOLD = 0.25
DEVICE = "mps"  # change to "cpu" if any issue


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT

    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    if not input_path.exists():
        print(f"❌ Input not found: {input_path}")
        return

    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print(f"Running inference on {input_path}")
    results = model.predict(
        source=str(input_path),
        conf=CONF_THRESHOLD,
        device=DEVICE,
        save=True,
        project=str(OUTPUT_DIR),
        name="run",
        exist_ok=True,
    )

    print("\n=== Detections summary ===")
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            dets = ", ".join(
                f"{model.names[int(c)]}({float(conf):.2f})"
                for c, conf in zip(r.boxes.cls, r.boxes.conf)
            )
            print(f"  {Path(r.path).name}: {dets}")
        else:
            print(f"  {Path(r.path).name}: no detections")

    print(f"\nAnnotated images: {OUTPUT_DIR}/run/")


if __name__ == "__main__":
    main()