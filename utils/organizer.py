import shutil
import os
from pathlib import Path

"replacing _combined.txt with _og.txt in labels folder"
def rename_images(folder_path):
    for img in folder_path.glob("*_combined.txt"):
        new_name = img.name.replace("_combined.txt", "_og.txt")
        new_path = img.parent / new_name
        img.rename(new_path)

folder_path = Path("D:/intezet/Bogi/Yolo/data_rcnn/labels")
# rename_images(folder_path)

"""Based on the annotated folder full of images, collect the images from the collected_images folder and copy them to the images folder based on these rules:
1. The image name must match the corresponding annotation file (without the _og.txt suffix).
2. For each image there are two matching files names in the collected folder, one with basic .png extension the other with _red.png extension
3. Copy both images to the images folder adding the _og.txt suffix to the basic image name and copy the _red image as is.
"""

def copy_images(annotated_folder, collected_images_folder, images_folder):
    os.makedirs(images_folder, exist_ok=True)
    
    for image_file in os.listdir(annotated_folder):
        
        base_name = image_file.replace("_combined.png", "")
        basic_image_name = f"{base_name}.png"
        red_image_name = f"{base_name}_red.png"
        
        basic_image_path = os.path.join(collected_images_folder, basic_image_name)
        red_image_path = os.path.join(collected_images_folder, red_image_name)
        
        if os.path.exists(basic_image_path):
            shutil.copy(basic_image_path, os.path.join(images_folder, f"{base_name}_og.png"))
            print(f"Copied {basic_image_name} to {images_folder}")
        
        if os.path.exists(red_image_path):
            shutil.copy(red_image_path, os.path.join(images_folder, red_image_name))
            print(f"Copied {red_image_name} to {images_folder}")
            
annotated_folder = "D:/intezet/Bogi/data/annotated"
collected_images_folder = "D:/intezet/Bogi/data/collected_images"
images_folder = "D:/intezet/Bogi/Yolo/data_rcnn/images"
    
# copy_images(annotated_folder, collected_images_folder, images_folder)


"""Remove every image that has the extension _red_red.png or _og.png in the given folder"""

def remove_images_with_suffix(folder_path, suffixes):
    for img in folder_path.glob("*"):
        if suffixes[0] in img.name or suffixes[1] in img.name:
            img.unlink()
            print(f"Removed {img.name}")
    
folder_path = Path("D:/intezet/Bogi/data/collected_images")
# remove_images_with_suffix(folder_path, ["_red_red.png", "_og.png"])


"""From the labels folder list out every txt file that is empty, then remove the images with the corresponding names from the images folder"""

def remove_empty_labels_and_images(label_folder, image_folder):
    for label_file in os.listdir(label_folder):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_folder, label_file)
            if os.path.getsize(label_path) == 0:
                image_name_base = label_file.replace("_og.txt", "_og.png")
                image_name_red = label_file.replace("_og.txt", "_red.png")
                image_path_base = os.path.join(image_folder, image_name_base)
                image_path_red = os.path.join(image_folder, image_name_red)
                if os.path.exists(image_path_base):
                    os.remove(image_path_base)
                    print(f"Removed image {image_name_base} corresponding to empty label {label_file}")
                if os.path.exists(image_path_red):
                    os.remove(image_path_red)
                    print(f"Removed image {image_name_red} corresponding to empty label {label_file}")
                os.remove(label_path)
                print(f"Removed empty label file {label_file}")
                    
label_folder = "D:/intezet/Bogi/Yolo/data_rcnn/labels"
image_folder = "D:/intezet/Bogi/Yolo/data_rcnn/images"
# remove_empty_labels_and_images(label_folder, image_folder)