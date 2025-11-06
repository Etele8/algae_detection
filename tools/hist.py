# gt_area_hist.py
from __future__ import annotations
import json, csv, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

# ----------------------------
# CONFIG — EDIT THESE PATHS
# ----------------------------
CONFIG = {
    "dataset": {
        # one of: "coco", "voc", "yolo", "csv"
        "format": "yolo",

        # COCO:
        "coco_json": "D:/data/annotations/instances_train.json",
        "coco_images_root": "D:/data/images/train",

        # VOC:
        "voc_xml_dir": "D:/data/VOC/Annotations",      # folder of .xml files
        "voc_images_root": "D:/data/VOC/JPEGImages",   # used if you want image-size sanity

        # YOLO (txt with normalized cx,cy,w,h per line):
        "yolo_labels_dir": "D:/intezet/Bogi/data/r-cnn/xml_labels",      # folder of .txt
        "yolo_images_root": "D:/intezet/Bogi/data/r-cnn/images",

        # CSV (columns: image_path,class,xmin,ymin,xmax,ymax) — header required
        "csv_file": "D:/data/boxes.csv",
    },

    "output_dir": "gt_area_hist_out",

    # how to label classes in plots/CSVs
    # leave empty for native names/ids; otherwise provide a mapping {id_or_name -> pretty_name}
    "class_name_map": {},

    # histogram binning
    "bins": {
        # "log10" → bins are evenly spaced in log10(area), good for wide range
        # "linear" → evenly spaced in area
        "mode": "log10",
        "n_bins": 60,
        # optional fixed range; leave None to auto from data
        "range": None,  # e.g., [2e2, 2e5]
    },

    # write per-class CSV of areas, and global summary JSON
    "write_csv": True,
    "write_json": True,

    # also compute side-length histogram (sqrt(area)) — handy for anchor sizing
    "plot_side_hist": True,
}

# ----------------------------
# Data containers
# ----------------------------
@dataclass
class Record:
    image_path: Path
    width: int
    height: int
    boxes_xyxy: np.ndarray  # (N,4) -> [xmin, ymin, xmax, ymax]
    labels: List[str]       # class names or ids as strings


# ----------------------------
# Loaders for various formats
# ----------------------------
def _area_xyxy(b: np.ndarray) -> np.ndarray:
    w = np.maximum(0.0, b[:, 2] - b[:, 0])
    h = np.maximum(0.0, b[:, 3] - b[:, 1])
    return w * h

def load_coco(ann_json: Path, images_root: Path, class_name_map: Dict = None) -> List[Record]:
    data = json.load(open(ann_json, "r", encoding="utf-8"))
    id_to_img = {im["id"]: im for im in data["images"]}
    id_to_cat = {c["id"]: c["name"] for c in data["categories"]}
    img_to_anns: Dict[int, List[dict]] = {}
    for a in data["annotations"]:
        if a.get("iscrowd", 0) == 1:  # skip crowd by default
            continue
        img_to_anns.setdefault(a["image_id"], []).append(a)
    recs: List[Record] = []
    for img_id, im in id_to_img.items():
        anns = img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(str(id_to_cat.get(a["category_id"], a["category_id"])))
        if not boxes:
            continue
        img_path = Path(images_root) / im["file_name"]
        recs.append(Record(
            image_path=img_path,
            width=int(im["width"]),
            height=int(im["height"]),
            boxes_xyxy=np.array(boxes, dtype=np.float32),
            labels=[class_name_map.get(l, l) if class_name_map else l for l in labels],
        ))
    return recs

def load_voc(xml_dir: Path, images_root: Optional[Path] = None, class_name_map: Dict = None) -> List[Record]:
    recs: List[Record] = []
    for xml_file in sorted(Path(xml_dir).glob("*.xml")):
        root = ET.parse(xml_file).getroot()
        fn = root.findtext("filename")
        size = root.find("size")
        w = int(size.findtext("width")); h = int(size.findtext("height"))
        boxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            bb = obj.find("bndbox")
            xmin = float(bb.findtext("xmin")); ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax")); ymax = float(bb.findtext("ymax"))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(str(name))
        if not boxes:
            continue
        if images_root is not None:
            img_path = Path(images_root) / fn
        else:
            img_path = Path(fn)
        recs.append(Record(
            image_path=img_path,
            width=w, height=h,
            boxes_xyxy=np.array(boxes, dtype=np.float32),
            labels=[class_name_map.get(l, l) if class_name_map else l for l in labels],
        ))
    return recs

def load_yolo(labels_dir: Path, images_root: Path, class_name_map: Dict = None) -> List[Record]:
    recs: List[Record] = []
    txts = sorted(Path(labels_dir).glob("*.txt"))
    for tf in txts:
        # infer image path by replacing .txt with common image suffix
        stem = tf.stem
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            p = Path(images_root) / f"{stem}{ext}"
            if p.exists():
                img_path = p; break
        if img_path is None:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = []
        labels = []
        for line in open(tf, "r", encoding="utf-8").read().strip().splitlines():
            if not line.strip():
                continue
            if len(line.split()) == 3:
                    continue
            parts = line.split()
            cls = parts[0]
            if cls == '8':
                cls = '4'
            elif cls == '7':
                continue
            cx, cy, ww, hh = map(float, parts[1:5])
            # YOLO -> pixel xyxy
            bw = ww * w; bh = hh * h
            x = cx * w - bw / 2.0
            y = cy * h - bh / 2.0
            boxes.append([x, y, x + bw, y + bh])
            labels.append(str(cls))
        if not boxes:
            continue
        recs.append(Record(
            image_path=img_path,
            width=w, height=h,
            boxes_xyxy=np.array(boxes, dtype=np.float32),
            labels=[class_name_map.get(l, l) if class_name_map else l for l in labels],
        ))
    return recs

def load_csv(csv_file: Path, class_name_map: Dict = None) -> List[Record]:
    rows = list(csv.DictReader(open(csv_file, "r", encoding="utf-8")))
    # group by image
    by_img: Dict[str, List[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image_path"], []).append(r)
    recs: List[Record] = []
    for ip, group in by_img.items():
        # try to get image size
        p = Path(ip)
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            # fallback: compute max coords
            w = h = None
        boxes = []
        labels = []
        for r in group:
            xmin = float(r["xmin"]); ymin = float(r["ymin"])
            xmax = float(r["xmax"]); ymax = float(r["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(str(r["class"]))
        if not boxes:
            continue
        if w is None or h is None:
            arr = np.array(boxes, dtype=np.float32)
            w = int(np.ceil(arr[:, [0, 2]].max()))
            h = int(np.ceil(arr[:, [1, 3]].max()))
        recs.append(Record(
            image_path=p,
            width=w, height=h,
            boxes_xyxy=np.array(boxes, dtype=np.float32),
            labels=[class_name_map.get(l, l) if class_name_map else l for l in labels],
        ))
    return recs

# ----------------------------
# Stats + plotting
# ----------------------------
def summarize_and_plot(recs: List[Record], out_dir: Path, bins_cfg: dict, class_name_map: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect areas per class
    per_class: Dict[str, List[float]] = {}
    overall: List[float] = []
    for r in recs:
        areas = _area_xyxy(r.boxes_xyxy)
        for a, lab in zip(areas, r.labels):
            per_class.setdefault(lab, []).append(float(a))
            overall.append(float(a))

    # basic stats
    def _stats(v: List[float]) -> Dict[str, float]:
        x = np.array(v, dtype=np.float64)
        if x.size == 0:
            return {"count": 0}
        return {
            "count": int(x.size),
            "min": float(np.min(x)),
            "p25": float(np.percentile(x, 25)),
            "p50": float(np.percentile(x, 50)),
            "p75": float(np.percentile(x, 75)),
            "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
        }

    summary = {"overall": _stats(overall)}
    for k, v in sorted(per_class.items(), key=lambda kv: kv[0]):
        pretty = class_name_map.get(k, k) if class_name_map else k
        summary[pretty] = _stats(v)

    # write JSON summary
    if CONFIG.get("write_json", True):
        with open(out_dir / "area_stats.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # write per-class CSV (flat list)
    if CONFIG.get("write_csv", True):
        with open(out_dir / "areas_per_class.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "area_px2"])
            for k, v in per_class.items():
                pretty = class_name_map.get(k, k) if class_name_map else k
                for a in v: w.writerow([pretty, int(round(a))])

    # plotting
    mode = bins_cfg.get("mode", "log10")
    n_bins = int(bins_cfg.get("n_bins", 60))
    rng = bins_cfg.get("range", None)

    def _make_bins(vals: np.ndarray):
        if mode == "log10":
            x = np.array(vals, dtype=np.float64)
            x = x[x > 0]
            if x.size == 0: return None, None, None
            lo = np.log10(np.min(x)) if rng is None else math.log10(rng[0])
            hi = np.log10(np.max(x)) if rng is None else math.log10(rng[1])
            edges = np.linspace(lo, hi, n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return np.histogram(np.log10(x), bins=edges), centers, "log10(area px^2)"
        else:
            x = np.array(vals, dtype=np.float64)
            if rng is None:
                lo, hi = np.min(x), np.max(x)
            else:
                lo, hi = rng
            edges = np.linspace(lo, hi, n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return np.histogram(x, bins=edges), centers, "area (px^2)"

    # overall histogram
    if len(overall):
        (counts, edges), centers, xlabel = _make_bins(overall)
        plt.figure(figsize=(9, 4))
        plt.bar(centers, counts, width=(centers[1]-centers[0]) if centers.size > 1 else 0.1, align="center")
        plt.title("GT area histogram (overall)")
        plt.xlabel(xlabel); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "hist_overall.png", dpi=160)
        plt.close()

    # per-class histograms
    for k, v in per_class.items():
        if not v: continue
        (counts, edges), centers, xlabel = _make_bins(v)
        plt.figure(figsize=(9, 4))
        plt.bar(centers, counts, width=(centers[1]-centers[0]) if centers.size > 1 else 0.1, align="center")
        plt.title(f"GT area histogram — class: {k}")
        plt.xlabel(xlabel); plt.ylabel("count")
        plt.tight_layout()
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(k))
        plt.savefig(out_dir / f"hist_class_{safe}.png", dpi=160)
        plt.close()

    # side-length hist (sqrt(area)) if requested
    if CONFIG.get("plot_side_hist", True) and len(overall):
        sides = np.sqrt(np.maximum(0.0, np.array(overall, dtype=np.float64)))
        plt.figure(figsize=(9, 4))
        plt.hist(sides, bins=60)
        plt.title("GT side-length histogram (sqrt(area)) — overall")
        plt.xlabel("side length (px)"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "hist_side_len_overall.png", dpi=160)
        plt.close()

    # quick console print
    print("\n[SUMMARY] area (px^2)")
    for k in ["overall"] + [kk for kk in summary.keys() if kk != "overall"]:
        s = summary[k]
        if not s or s["count"] == 0:
            print(f"  {k:>12}: (no boxes)")
        else:
            print(f"  {k:>12}: n={s['count']}, p50={s['p50']:.1f}, p90={s['p90']:.1f}, mean={s['mean']:.1f}")

# ----------------------------
# Main
# ----------------------------
def main():
    cfg = CONFIG
    ds = cfg["dataset"]
    fmt = ds["format"].lower().strip()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    class_map = cfg.get("class_name_map", {}) or {}

    if fmt == "coco":
        recs = load_coco(Path(ds["coco_json"]), Path(ds["coco_images_root"]), class_map)
    elif fmt == "voc":
        recs = load_voc(Path(ds["voc_xml_dir"]), Path(ds["voc_images_root"]), class_map)
    elif fmt == "yolo":
        recs = load_yolo(Path(ds["yolo_labels_dir"]), Path(ds["yolo_images_root"]), class_map)
    elif fmt == "csv":
        recs = load_csv(Path(ds["csv_file"]), class_map)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if not recs:
        print("No annotations found.")
        return

    summarize_and_plot(recs, out_dir, cfg["bins"], class_map)
    print(f"\n[OK] Results in: {out_dir}")

if __name__ == "__main__":
    main()
