import os
import glob

"""Creating annotated.txt file from the names of the images in the labels folder"""
label_dir = "D:/intezet/Bogi/Yolo/data_rcnn/labels"
annotated_file_path = "D:/intezet/Bogi/data/annotated.txt"

with open(annotated_file_path, "w") as annotated_file:
    for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
        image_file_name = os.path.basename(label_file).replace("_og.txt", "")
        annotated_file.write(image_file_name + "\n")

"""Getting the names form annotated.txt, moving those from the merged folder to annotated folder"""

annotated_file_path = "D:/intezet/Bogi/data/annotated.txt"
merged_path = "D:/intezet/Bogi/data/merged"
annotated_folder_path = "D:/intezet/Bogi/data/annotated"
os.makedirs(annotated_folder_path, exist_ok=True)
for image_file_name in open(annotated_file_path, "r"):
    image_file_name = image_file_name.strip() 
    image_file_name += "_combined.png"
    if not image_file_name:
        continue
    # Check if the file exists in the merged folder
    merged_image_path = os.path.join(merged_path, image_file_name)
    if os.path.exists(merged_image_path):
        # Move the file to the annotated folder
        os.rename(merged_image_path, os.path.join(annotated_folder_path, image_file_name))
        print(f"Moved {image_file_name} to {annotated_folder_path}")
    else:
        print(f"{image_file_name} not found in {merged_path}")