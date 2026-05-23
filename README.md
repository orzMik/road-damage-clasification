# Road Damage Classification

Project for Neural Networks Course (2026). The goal is to detect and classify road damage while focusing on reducing false positive detections (e.g., distinguishing potholes from manholes).

## Project Overview
* **Theme**: Automatic detection and classification of road damage (potholes and cracks) using images from low-cost cameras (smartphones, action cams).
* **Innovation**: Many existing systems struggle with false positives, often misidentifying manholes as potholes. Our model is trained to explicitly distinguish between actual damage and standard road infrastructure.
* **Dataset**: A real-world dataset collected in Rome, containing 2009 images with three classes: `0 - Pothole`, `1 - Crack`, and `2 - Manhole`.

## Technology Stack
* **Architecture**: Modern YOLO-family models (e.g., YOLOv8) and potentially RFDetr.
* **Augmentation**: `albumentations` library for robustness against lighting conditions and noise.
* **Framework**: Ultralytics/PyTorch Lightning for training and Weights & Biases for experiment tracking.
* **Environment Management**: `uv` for high-performance dependency resolution.

## Project Management

* **[TODO & Milestones](todo.md)**: Track our progress and upcoming deadlines.
* **[Setup & Installation](setup.md)**: Instructions on how to set up the environment, download data, and configure W&B.
* **[Scripts](scripts.md)**: List of working scripts in our project.

## Project Structure
* `data/`: Datasets (ignored by git).
* `notebooks/`: Experimental notebooks for EDA and testing.
* `scripts/`: Production training and evaluation scripts.
* `src/`: Core source code including models, datasets, and configurations.
* `pyproject.toml`: Main project configuration and dependency management.

## Google Colab Workflow
* **`notebooks/colab_template.ipynb`** — bootstrap only: clone repo, mount Drive, resolve paths (`data/`, `models/`, `runs/`, `wroclaw_images/`).
* **`notebooks/yolo_v8_baseline.ipynb`** — baseline YOLOv8 experiment: train, test evaluation, inference on `wroclaw_images/`.

Helpers live under `src/config/colab/drive.py`, `src/config/wandb.py`, and `src/config/yolo.py` (also re-exported from `src/colab`).

Set `DRIVE_ROOT` in the notebook to match your Drive location:
* My Drive: `MyDrive/road-damage-detection`
* Shared drive: `Shareddrives/<TeamDrive>/road-damage-detection`

## Team
* Wiktor Małysa
* Hubert Stolarz
* Karol Kulig
* Mikołaj Orzechowski
