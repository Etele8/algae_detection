"""Stage 5 - Review: human-in-the-loop outputs.

For each discovered cluster we render a contact sheet (a grid of organism
thumbnails) so the researcher can recognise the morphotype at a glance and give
the whole group one taxonomic name - instead of labelling thousands of cells.
We also emit a 2-D scatter of all objects and a ``cluster_labels.csv`` template
to fill in. Those names feed the classifier stage.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import Config


def _thumb(raw_path: str, size: int) -> np.ndarray:
    img = cv2.imread(raw_path)
    if img is None:
        return np.full((size, size, 3), 200, np.uint8)
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _contact_sheet(paths: list[str], size: int) -> np.ndarray:
    n = len(paths)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    pad = 4
    sheet = np.full((rows * (size + pad) + pad, cols * (size + pad) + pad, 3), 30, np.uint8)
    for i, p in enumerate(paths):
        r, cc = divmod(i, cols)
        y = pad + r * (size + pad)
        x = pad + cc * (size + pad)
        sheet[y:y + size, x:x + size] = _thumb(p, size)
    return sheet


def run_review(cfg: Config) -> Path:
    clusters = pd.read_csv(cfg.outputs_dir / "clusters.csv")
    objects = pd.read_csv(cfg.outputs_dir / "objects.csv")
    df = clusters.merge(objects[["object_id", "raw_path"]], on="object_id", how="left")

    r = cfg["review"]
    size = int(r["thumb_size"])
    per = int(r["thumbs_per_cluster"])

    sheets_dir = cfg.outputs_dir / "clusters"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for cl, grp in df.groupby("cluster"):
        # Show the objects nearest the cluster's 2-D centroid first (most typical).
        cx, cy = grp["x"].mean(), grp["y"].mean()
        order = ((grp["x"] - cx) ** 2 + (grp["y"] - cy) ** 2).sort_values().index
        paths = grp.loc[order, "raw_path"].head(per).tolist()
        sheet = _contact_sheet(paths, size)
        name = "noise" if cl == -1 else f"cluster_{int(cl):02d}"
        cv2.imwrite(str(sheets_dir / f"{name}.png"), sheet)
        summary.append({"cluster": int(cl), "n_objects": int(len(grp)), "sheet": f"clusters/{name}.png"})

    summary_df = pd.DataFrame(summary).sort_values("cluster")
    summary_df.to_csv(cfg.outputs_dir / "cluster_summary.csv", index=False)

    # Labeling template: researcher fills the `label` column once per cluster.
    template = cfg.outputs_dir / "cluster_labels.csv"
    if not template.exists():
        label_df = summary_df[["cluster", "n_objects"]].copy()
        label_df["label"] = ""   # e.g. "Dolichospermum", "filamentous cyanobacteria", ...
        label_df["notes"] = ""
        label_df.to_csv(template, index=False)

    _scatter(df, cfg.outputs_dir / "scatter.png")
    print(f"Wrote {len(summary)} contact sheets to {sheets_dir} and labeling "
          f"template -> {template}")
    return sheets_dir


def _scatter(df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    for cl, grp in df.groupby("cluster"):
        label = "noise" if cl == -1 else f"c{int(cl)}"
        ax.scatter(grp["x"], grp["y"], s=14, alpha=0.7, label=label)
    ax.set_title("Organism embeddings (2-D projection), coloured by morphotype cluster")
    ax.legend(markerscale=1.5, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
