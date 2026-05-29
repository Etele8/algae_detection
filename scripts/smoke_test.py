"""End-to-end correctness smoke test on a small image subset (CPU-friendly).

Runs the whole pipeline on 3 frames with relaxed clustering, auto-fills labels
to exercise train/predict, and asserts every expected artifact exists. This
proves the code is correct without paying for a full CPU SAM run.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
os.environ.setdefault("YOLO_CONFIG_DIR", "models")

import pandas as pd  # noqa: E402

from algae import classify, cluster, detect, embed, manifest, review  # noqa: E402
from algae.config import load_config  # noqa: E402

cfg = load_config()
# Relax for a tiny subset so the chain produces real clusters, not all-noise.
cfg.raw["detect"]["max_images"] = 3
cfg.raw["cluster"]["hdbscan"]["min_cluster_size"] = 3

manifest.write_manifest(cfg)
detect.run_detection(cfg)
embed.run_embedding(cfg)
cluster.run_clustering(cfg)
review.run_review(cfg)

# Auto-fill labels just to exercise the supervised tail end.
lp = cfg.outputs_dir / "cluster_labels.csv"
df = pd.read_csv(lp)
df["label"] = ["noise" if c == -1 else f"type_{int(c)}" for c in df["cluster"]]
df.to_csv(lp, index=False)

classify.train_classifier(cfg)
classify.label_all_objects(cfg)

expected = [
    "manifest.csv", "objects.csv", "embeddings.npy", "embeddings_index.csv",
    "clusters.csv", "cluster_summary.csv", "cluster_labels.csv",
    "scatter.png", "classifier.joblib", "predictions.csv",
]
missing = [f for f in expected if not (cfg.outputs_dir / f).exists()]
sheets = list((cfg.outputs_dir / "clusters").glob("*.png"))
print("MISSING:", missing)
print("CONTACT SHEETS:", len(sheets))
print("SMOKE OK" if not missing and sheets else "SMOKE FAILED")
