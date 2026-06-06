# TODO & Milestones

## ✅ Already Done
* [x] Define project scope and problem (reducing false positives).
* [x] Select initial dataset (Kaggle - 2009 images from Rome).
* [x] Setup environment and dependency management (`uv`, `pyproject.toml`).
* [x] Create core project structure, environment checkers, and dataset download scripts.
* [x] Initial Exploratory Data Analysis (EDA) and draft training notebooks.

## ⏳ Ongoing Tasks (Target: First Progress Meeting)

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

## New
Rf-detr - do we really need
- podsumowanie stattytyk datasetu, ile jest danych klas w zbiorze treningowym
