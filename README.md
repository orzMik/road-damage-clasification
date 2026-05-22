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
Use `notebooks/colab_template.ipynb` as the starting point for training, evaluation, and inference in Colab. It clones the repo, mounts Google Drive, and calls helpers from `src/colab/yolo.py`.

Set `DriveConfig.drive_root` to a shared location such as `Shareddrives/<shared_drive_name>/road-damage` to keep datasets, runs, and model artifacts in one shared folder.

## Team
* Wiktor Małysa
* Hubert Stolarz
* Karol Kulig
* Mikołaj Orzechowski
