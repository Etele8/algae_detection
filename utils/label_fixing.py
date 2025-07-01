import os

# Paths
label_dir = "D:/intezet/Bogi/Yolo/data/labels_old/val"  # change if needed
output_dir = "D:/intezet/Bogi/Yolo/data/labels/val"  # where to save updated labels
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(label_dir):
    if not fname.endswith(".txt"):
        continue
    src = os.path.join(label_dir, fname)
    dst = os.path.join(output_dir, fname)

    with open(src, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip bad lines
        cls, x, y, w, h = map(float, parts)
        x = x * 2
        w = w * 2
        new_lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(dst, "w") as f:
        f.write("\n".join(new_lines))