## Environment setup

### 1. Clone the Repository
First, clone the project repository to your local machine and navigate into the project directory:
```bash
git clone [https://github.com/orzmik/road-damage-detection.git](https://github.com/orzmik/road-damage-detection.git)

cd road-damage-detection
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

## Weights & Biases Setup

We use W&B for experiment tracking and team collaboration.

1. Get your API key: [wandb.ai/authorize](https://wandb.ai/authorize).
2. Add your W&B configuration to the `.env` file:

```text
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=road-damage-classification
WANDB_ENTITY=project-nn
```

3. The training scripts will automatically detect these variables and sync results to the team workspace.

## Google Colab

Use `notebooks/colab_template.ipynb` as the shared starting point for Colab runs. It mounts Google Drive, configures a shared folder via `DriveConfig.drive_root`, and uses `src/colab/yolo.py` for training, evaluation, and inference with Ultralytics + W&B.
