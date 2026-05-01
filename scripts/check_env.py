import torch
import ultralytics
import albumentations as A

def check_environment():
    print("--- Environment Check ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"Albumentations version: {A.__version__}")
    print("-------------------------")

    if torch.cuda.is_available():
        print("Hardware Acceleration: NVIDIA CUDA is AVAILABLE.")
    elif torch.backends.mps.is_available():
        print("Hardware Acceleration: Apple Silicon MPS is AVAILABLE.")
    else:
        print("Hardware Acceleration: NONE. Running on CPU.")

if __name__ == "__main__":
    check_environment()