import os, xml.etree.ElementTree as ET
from pathlib import Path

# ---- CONFIG ----
ROOT = Path(__file__).resolve().parent.parent
INPUT_XML = ROOT / "data" / "r-cnn" / "xmls" / "5.xml"        # path to your CVAT XML
OUT_DIR   = ROOT / "data" / "r-cnn" / "xml_labels"             # one TXT per image here
WRITE_EMPTY_FILES = True                                       # create empty .txt if image has no annos

# 0-based class ids — adjust if needed
CLASS2ID = {
    "EUK": 0,
    "FE": 1,
    "FC": 2,
    "EUK colony": 3,
    "FE colony": 4,
    "FC colony": 5,
    "colony_cell": 6,
    "crowd": 7,
    "high_overlap": 8,
}

def norm(v, d): return float(v) / float(d) if d else 0.0

OUT_DIR.mkdir(parents=True, exist_ok=True)
root = ET.parse(INPUT_XML).getroot()
images = root.findall(".//image")

n_boxes = n_points = 0
for im in images:
    name = im.attrib["name"]
    W = float(im.attrib["width"]); H = float(im.attrib["height"])
    base = os.path.splitext(os.path.basename(name))[0]
    lines = []

    # Boxes -> YOLO (class cx cy w h)
    for box in im.findall("./box"):
        label = box.attrib.get("label")
        if label not in CLASS2ID: continue
        cls = CLASS2ID[label]
        xtl = float(box.attrib["xtl"]); ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"]); ybr = float(box.attrib["ybr"])
        cx = (xtl + xbr) / 2.0; cy = (ytl + ybr) / 2.0
        bw = max(0.0, xbr - xtl); bh = max(0.0, ybr - ytl)
        lines.append(f"{cls} {norm(cx,W):.6f} {norm(cy,H):.6f} {norm(bw,W):.6f} {norm(bh,H):.6f}")
        n_boxes += 1

    # Points -> (class x y)
    for pt in im.findall("./points"):
        label = pt.attrib.get("label")
        if label not in CLASS2ID: continue
        cls = CLASS2ID[label]
        x_str, y_str = pt.attrib["points"].split(",")
        x = float(x_str); y = float(y_str)
        lines.append(f"{cls} {norm(x,W):.6f} {norm(y,H):.6f}")
        n_points += 1

    # Write one file per image
    p = OUT_DIR / f"{base}.txt"
    if lines or WRITE_EMPTY_FILES:
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

print(f"Done. Boxes: {n_boxes}, Points: {n_points}. Output dir: {OUT_DIR}")
