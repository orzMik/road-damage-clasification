# Road Damage Classification

Project for Neural Networks Course (2026). The goal is to detect and classify road damage while focusing on reducing false positive detections (e.g., distinguishing potholes from manholes).

## Project Overview
* **Theme**: Automatic detection and classification of road damage (potholes and cracks) using images from low-cost cameras (smartphones, action cams).
* **Innovation**: Many existing systems struggle with false positives, often misidentifying manholes as potholes. Our model is trained to explicitly distinguish between actual damage and standard road infrastructure.
* **Dataset**: A real-world dataset collected in Rome, containing 2009 images with three classes: `0 - Pothole`, `1 - Crack`, and `2 - Manhole`.

## Technology Stack
* **Architecture**: Modern YOLO-family models (e.g., YOLOv8) and potentially RFDetr.
* **Augmentation**: `albumentations` library for robustness against lighting conditions and noise.
* **Framework**: PyTorch Lightning for training and Weights & Biases for experiment tracking.
* **Environment Management**: `uv` for high-performance dependency resolution.

## TODO & Milestones

### ✅ Already Done
* [x] Define project scope and problem (reducing false positives).
* [x] Select initial dataset (Kaggle - 2009 images from Rome).
* [x] Setup environment and dependency management (`uv`, `pyproject.toml`).
* [x] Create core project structure, environment checkers, and dataset download scripts.
* [x] Initial Exploratory Data Analysis (EDA) and draft training notebooks.

### ⏳ Ongoing Tasks (Target: First Progress Meeting)

**Data Preparation:**
* [ ] Collect our own real-world photos of streets (potholes, cracks, and manholes).
* [ ] Annotate the custom images and merge them with the base Kaggle dataset to improve model robustness.
* [ ] Verify the correctness of the experimental setup (ensure proper train/val/test splits to avoid data leakage).

**Pipeline & Baseline Implementation:**
* [ ] Formulate the precise ML task and define evaluation metrics (e.g., mAP, with a special focus on tracking false positives for the manhole class).
* [ ] Implement an end-to-end baseline training pipeline using PyTorch Lightning.
* [ ] Integrate data augmentations (`albumentations`) into the data loaders.
* [ ] Train the baseline solution (e.g., standard YOLOv8) on the initial dataset.
* [ ] Connect Weights & Biases to log initial experiments and validate that the whole pipeline is working correctly.

### Project Topic Meeting (March 31) - *6 points*
* [x] Prepare presentation covering the problem to solve and planned methodology.
* [x] Ensure clarity of the problem definition and suitability of the chosen dataset.
* [x] Validate the feasibility of the project scope and the quality of the proposed approach.

### First Progress Meeting (April 28 / May 12) - *12 points*
* [ ] Correctly formulate the ML task.
* [ ] Implement a baseline solution.
* [ ] Conduct initial experiments and ensure the correctness of the experimental setup.
* [ ] Demonstrate that the end-to-end project pipeline is working.

## Setup Instructions

### 1. Clone the Repository
First, clone the project repository to your local machine and navigate into the project directory:
```bash
git clone [https://github.com/orzmik/road-damage-clasification.git](https://github.com/orzmik/road-damage-clasification.git)
cd road-damage-clasification

```

### 2. Install uv
* **Windows (PowerShell)**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
* **macOS**: `brew install uv`

### 3. Initialize Environment
Run these commands in the project root directory:
```bash
# Create virtual environment with Python 3.12
uv venv --python 3.12

# Activate environment
# Windows:
.venv\\Scripts\\activate
# macOS:
source .venv/bin/activate

# Sync dependencies (detects hardware automatically based on pyproject.toml)
uv sync
```

### 4. Verify Environment
To ensure your hardware acceleration (CUDA/MPS) is correctly detected, run:
```bash
uv run scripts/check_env.py
```

## Dataset Download

The dataset is hosted on Kaggle. To download it automatically, each team member must use their own Kaggle API token.

### 1. Get Kaggle Credentials

Go to your Kaggle account **Settings** -> **API tokens** -> **Generate New Token**.


### 2. Configure Local Environment
1. Create a file named `.env` in the root directory of the project.
2. Add your Kaggle credentials to this file exactly in this format:
```text
KAGGLE_USERNAME=your_username_here
KAGGLE_KEY=your_api_key_here
```

### 3. Run the Download Script
Execute the following command to download and extract the data:
```bash
uv run scripts/download_dataset.py
```


## Project Structure
* `data/`: Datasets (ignored by git).
* `notebooks/`: Experimental notebooks for EDA and testing.
* `scripts/`: Production training and evaluation scripts.
* `src/`: Core source code including models, datasets, and configurations.
* `pyproject.toml`: Main project configuration and dependency management.

## Team
* Wiktor Małysa
* Hubert Stolarz
* Karol Kulig
* Mikołaj Orzechowski