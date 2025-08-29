import os
import shutil
from pathlib import Path



"""
Run after every annotation session to clean up and organize the dataset for training.
This script performs the following tasks:
1. Renames label files from *_combined.txt to *_og.txt.
2. Moves annotated images from the merged folder to the annotated folder.
3. Copies original and red images to the training images folder.
4. Removes empty label files and their corresponding images.
"""


# === Paths ===
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MERGED_PATH = DATA / "merged"
ANNOTATED_FOLDER = DATA / "annotated"
COLLECTED_IMAGES = DATA / "collected_images"
RCNN_PATH = DATA / "r-cnn"
RCNN_TRAINING_IMAGES = RCNN_PATH / "images"
RCNN_LABELS = RCNN_PATH / "labels"


def get_annotated_names():
    
    """Get base filenames from label directory."""
    
    return [path.name.replace("_og.txt", "") for path in RCNN_LABELS.glob("*.txt")]


def move_annotated_images(annotated_names):
    
    """Move annotated images from merged to annotated folder."""
    
    ANNOTATED_FOLDER.mkdir(exist_ok=True)
    for name in annotated_names:
        img_name = name.replace("txt", "png")
        src = MERGED_PATH / img_name
        dst = ANNOTATED_FOLDER / img_name
        if src.exists():
            src.rename(dst)
            print(f"Moved {img_name} to {ANNOTATED_FOLDER}")
        else:
            print(f"{img_name} not found in {MERGED_PATH}")


def correct_combined_labels():
    
    """Double x and w values for combined labels to correct for concatenated image layout."""
    
    for path in RCNN_LABELS.glob("*_combined.txt"):
        with path.open("r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            new_lines.append(f"{int(cls)} {x*2:.6f} {y:.6f} {w*2:.6f} {h:.6f}")

        path.write_text("\n".join(new_lines))


def rename_combined_to_og():
    
    """Rename *_combined.txt label files to *_og.txt."""
    
    for path in RCNN_LABELS.glob("*_combined.txt"):
        new_name = path.name.replace("_combined.txt", "_og.txt")
        path.rename(path.parent / new_name)


def copy_annotated_images(annotated_names):
    
    """Copy original and red images for each annotated name to RCNN images folder."""
    
    RCNN_TRAINING_IMAGES.mkdir(parents=True, exist_ok=True)
    for name in annotated_names:
        original = COLLECTED_IMAGES / f"{name}.png"
        red = COLLECTED_IMAGES / f"{name}_red.png"

        if original.exists():
            shutil.copy(original, RCNN_TRAINING_IMAGES / f"{name}_og.png")
            print(f"Copied {original.name}")
        if red.exists():
            shutil.copy(red, RCNN_TRAINING_IMAGES / red.name)
            print(f"Copied {red.name}")


def remove_empty_labels_and_images():
    
    """Remove empty label files and their corresponding images."""
    
    for label in RCNN_LABELS.glob("*.txt"):
        if label.stat().st_size == 0:
            base = label.name.replace("_og.txt", "")
            og_img = RCNN_TRAINING_IMAGES / f"{base}_og.png"
            red_img = RCNN_TRAINING_IMAGES / f"{base}_red.png"

            for img in [og_img, red_img]:
                if img.exists():
                    img.unlink()
                    print(f"Removed image {img.name}")

            label.unlink()
            print(f"Removed empty label {label.name}")


def rename():
    """Rename .png image files without the _red ing file name to *_og.png."""
    for path in COLLECTED_IMAGES.glob("*.png"):
        if "_red" not in path.name:
            new_name = path.name.replace(".png", "_og.png")
            path.rename(path.parent / new_name)

def run_pipeline():
    # 1. Fix and rename label files
    correct_combined_labels()
    rename_combined_to_og()

    # 2. Get names of annotated stems
    annotated_names = get_annotated_names()

    # 3. Move annotated images
    move_annotated_images(annotated_names)

    # 4. Copy paired OG + RED images into training folder
    copy_annotated_images(annotated_names)

    # 5. Cleanup: remove empty labels/images
    remove_empty_labels_and_images()



if __name__ == "__main__":
    run_pipeline()