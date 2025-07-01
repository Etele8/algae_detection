import shutil
import os
from pathlib import Path

"add to every txt file names _og"
def rename_images(folder_path):
    for img in folder_path.glob("*.txt"):
        new_name = img.stem + "_og.txt"
        new_path = img.parent / new_name
        img.rename(new_path)

folder_path = Path("D:/intezet/Bogi/Yolo/data_rcnn/labels")
rename_images(folder_path)