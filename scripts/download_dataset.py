import os
from pathlib import Path
from dotenv import load_dotenv

def download_dataset():
    # load environment variables from .env file
    load_dotenv()

    dataset_slug = "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
    target_dir = Path(".")

    # verify kaggle credentials
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        print("ERROR: missing KAGGLE_USERNAME or KAGGLE_KEY in .env file.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory prepared at: {target_dir.resolve()}")

    # import kaggle library only after environment variables are loaded
    import kaggle

    print("Authenticating with Kaggle API...")
    kaggle.api.authenticate()

    print(f"Downloading and extracting dataset: {dataset_slug}...")
    kaggle.api.dataset_download_files(
        dataset_slug,
        path=str(target_dir),
        unzip=True
    )
    
    print("Download complete. Dataset is ready.")

if __name__ == "__main__":
    download_dataset()