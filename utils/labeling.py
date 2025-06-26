import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random

limit = 5

base_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(base_dir, "collected_images")

# Load normal images (skip "_red")
normal_images = sorted(glob.glob(os.path.join(image_folder, "Image_*.png")))
normal_images = [img for img in normal_images if "_red" not in img]
random.shuffle(normal_images)

# Resize helper
def resize_image(image, max_width=1200, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale

# Click callback
def show_hsv_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        original_img, scale = param
        x_orig = int(x / scale)
        y_orig = int(y / scale)
        pixel = original_img[y_orig, x_orig]
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_RGB2HSV)[0][0]
        print(f"Clicked HSV (at {x_orig},{y_orig}): H={hsv_pixel[0]}, S={hsv_pixel[1]}, V={hsv_pixel[2]}")

# Main loop
i = 0
for normal_path in normal_images:
    if i >= limit:
        break

    red_path = normal_path.replace(".png", "_red.png")
    if not os.path.exists(red_path):
        print(f"Missing: {red_path}")
        continue

    # Load and convert
    normal = cv2.imread(normal_path)
    red = cv2.imread(red_path)
    normal_rgb = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
    red_rgb = cv2.cvtColor(red, cv2.COLOR_BGR2RGB)
    hsv_normal = cv2.cvtColor(normal, cv2.COLOR_BGR2HSV)
    hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

    # === Thresholding ===
    bright_center_mask = cv2.inRange(hsv_normal, (0, 0, 200), (180, 80, 255))  # low S, high V
    yellow_mask_raw = cv2.inRange(hsv_normal, (20, 100, 200), (38, 255, 255))
    yellow_combined = cv2.bitwise_or(yellow_mask_raw, bright_center_mask)
    
    # close yellow mask to fill small holes
    kernel = np.ones((3, 3), np.uint8)
    yellow_combined = cv2.morphologyEx(yellow_combined, cv2.MORPH_CLOSE, kernel)

    red_mask1 = cv2.inRange(hsv_normal, (0, 100, 100), (19, 255, 255))
    red_mask2 = cv2.inRange(hsv_normal, (160, 100, 200), (180, 255, 255))
    red_mask_norm = cv2.bitwise_or(red_mask1, red_mask2)

    red_mask_enh1 = cv2.inRange(hsv_red, (0, 0, 100), (32, 255, 255))
    red_mask_enh2 = cv2.inRange(hsv_red, (160, 0, 100), (180, 255, 255))
    red_mask_enh = cv2.bitwise_or(red_mask_enh1, red_mask_enh2)

    # Combine red + bright center
    red_mask_combined = cv2.bitwise_or(red_mask_norm, bright_center_mask)

    # === Post-processing ===

    # Fill small holes in red mask
    red_mask_closed = cv2.morphologyEx(red_mask_combined, cv2.MORPH_CLOSE, kernel)

    # Find red contours
    contours, _ = cv2.findContours(red_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_cleaned = yellow_combined.copy()

    for cnt in contours:
        mask = np.zeros_like(red_mask_closed)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        yellow_cleaned[mask == 255] = 0  # Remove yellow inside red
        red_mask_closed[mask == 255] = 255  # Keep red mask intact
    
    kernel = np.ones((5, 5), np.uint8)
    red_mask_opened = cv2.morphologyEx(red_mask_closed, cv2.MORPH_OPEN, kernel)
    yellow_cleaned = cv2.morphologyEx(yellow_cleaned, cv2.MORPH_OPEN, kernel)

    # === Visualization ===
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(normal_rgb)
    plt.title("Normal Image"); plt.axis("off")
    plt.subplot(2, 3, 2)
    plt.imshow(normal_rgb)
    plt.title("Normal Image"); plt.axis("off")
    plt.subplot(2, 3, 3)
    plt.imshow(red_rgb, cmap='gray')
    plt.title("Red Image"); plt.axis("off")
    plt.subplot(2, 3, 4)
    plt.imshow(yellow_cleaned, cmap='gray')
    plt.title("Yellow Mask (Cleaned FE)"); plt.axis("off")
    plt.subplot(2, 3, 5)
    plt.imshow(red_mask_opened, cmap='gray')
    plt.title("Red Mask (EUK+FC, filled)"); plt.axis("off")
    plt.subplot(2, 3, 6)
    plt.imshow(red_mask_enh, cmap='gray')
    plt.title("Red Enhanced (FC)"); plt.axis("off")
    plt.tight_layout()
    plt.show()

    """# === HSV Inspector ===
    for rgb_img, label in [(normal_rgb, "HSV Inspector"), (red_rgb, "HSV Inspector red")]:
        resized_rgb, scale = resize_image(rgb_img)
        cv2.namedWindow(label, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(label, show_hsv_click, param=(rgb_img, scale))
        cv2.imshow(label, cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR))
        print(f"{label} - Click to see HSV. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        # === Final Classification & Labeling ===
    final_image = normal_rgb.copy()
    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

    def draw_and_label(mask, color, label, condition_mask=None):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000 or area < 20:  # Filter out algae and noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            if condition_mask is not None:
                submask = condition_mask[y:y+h, x:x+w]
                mask_region = mask[y:y+h, x:x+w]
                if np.count_nonzero(cv2.bitwise_and(submask, mask_region)) == 0:
                    continue  # Condition not met

            cv2.drawContours(final_image, [cnt], -1, color, 2)
            cv2.putText(final_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FE: Yellow only
    draw_and_label(yellow_cleaned, (0, 255, 255), "FE", condition_mask=cv2.bitwise_not(red_mask_opened))

    # FC: Red present in both red_mask_opened and red_mask_enh
    fc_overlap = cv2.bitwise_and(red_mask_opened, red_mask_enh)
    draw_and_label(fc_overlap, (0, 0, 255), "FC")

    # EUK: Red present in red_mask_opened but NOT in red_mask_enh
    euk_mask = cv2.bitwise_and(red_mask_opened, cv2.bitwise_not(red_mask_enh))
    draw_and_label(euk_mask, (255, 0, 0), "EUK")

    # === Show Labeled Output ===
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title("Final Classification")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    i += 1