Below is a list of available scripts in the `scripts/` directory and their purposes:

### Environment & Data
* **`scripts/check_env.py`**: Verifies that the virtual environment is correctly set up and checks for hardware acceleration (CUDA for NVIDIA or MPS for macOS).
```bash
uv run scripts/check_env.py
```

* **`scripts/download_dataset.py`**: Downloads and extracts the road damage dataset from Kaggle. Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` in the `.env` file.
```bash
uv run scripts/download_dataset.py
```

### Preprocessing

* **`scripts/split_data_yolo.py`**: Processes raw data into the YOLO-ready structure. It splits images and labels into `train`, `val`, and `test` sets within `data/processed-yolo/` and prints class distribution statistics.
```bash
uv run scripts/split_data_yolo.py
```

### Training

* **`scripts/train_baseline_yolo.py`**: Runs the baseline training using YOLOv8n (Nano). It initializes experiment tracking with Weights & Biases and uses early stopping (patience=8). Requires `WANDB_API_KEY`, `WANDB_PROJECT` and `WANDB_ENTITY` in `.env` file
```bash
uv run scripts/train_baseline_yolo.py
```