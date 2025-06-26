import os
import cv2
import glob
import numpy as np

# === CONFIG ===
input_dir = "collected_images"  # folder with Image_*.png and Image_*_red.png
output_dir = "merged_channels"  # folder to save combined images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get normal images (without _red)
normal_paths = sorted([p for p in glob.glob(os.path.join(input_dir, "Image_*.png")) if "_red" not in p])

count = 0
for normal_path in normal_paths:
    if count < 638:
        count += 1
        continue  # Skip first 638 images
    base_name = os.path.basename(normal_path)
    red_path = normal_path.replace(".png", "_red.png")
    
    if not os.path.exists(red_path):
        print(f"Skipping {base_name} (missing red version)")
        continue

    # Load images
    normal = cv2.imread(normal_path)
    red = cv2.imread(red_path)

    # Ensure they have the same size
    if normal.shape != red.shape:
        print(f"Resizing red to match normal for {base_name}")
        red = cv2.resize(red, (normal.shape[1], normal.shape[0]))

    # Concatenate side by side
    # combined = cv2.hconcat([normal, red])

    # Combine into 3 channels: original for R, enhanced for G, duplicate enhanced for B
    combined_img = np.zeros_like(normal)
    combined_img[:, :, 0] = normal[:, :, 0]  # Red channel from original
    combined_img[:, :, 1] = red[:, :, 1]  # Green channel from enhanced
    combined_img[:, :, 2] = red[:, :, 2]  # Blue channel from enhanced

    # Save the combined image
    output_path = os.path.join(output_dir, base_name.replace(".png", "_combined.png"))
    cv2.imwrite(output_path, combined_img)
    print(f"Saved combined image: {output_path}")
    # Save combined image
    count += 1
    print(f"{count} / {len(normal_paths)}")

print(f"✅ Done. {count} combined images saved to '{output_dir}'")
