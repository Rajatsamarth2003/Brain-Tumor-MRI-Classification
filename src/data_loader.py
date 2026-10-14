import os
from pathlib import Path

DATASET_DIR = Path("Datasets")


def show_dataset_structure():
    print("Dataset Structure")
    print("------------------")

    for folder in DATASET_DIR.iterdir():
        if folder.is_dir():
            print(f"\n{folder.name}")

            for subfolder in folder.iterdir():
                if subfolder.is_dir():
                    print(f"  └── {subfolder.name}")


if __name__ == "__main__":
    show_dataset_structure()