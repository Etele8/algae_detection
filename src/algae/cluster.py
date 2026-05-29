"""Stage 4 - Clustering: group organism embeddings into morphotypes.

HDBSCAN is the default because it finds clusters of varying density and, unlike
k-means, does not force every object into a group - rare/odd organisms land in
a noise bucket (label ``-1``) the researcher can inspect separately. A 2-D
projection is also saved for the scatter plot in the review stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .config import Config


def _reduce(x: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (features_for_clustering, coords_2d_for_plotting)."""
    c = cfg["cluster"]
    n_comp = min(int(c["pca_components"]), x.shape[0], x.shape[1])
    reduced = PCA(n_components=n_comp, random_state=cfg["seed"]).fit_transform(x)

    if c["reducer"] == "umap":
        try:
            import umap
            coords = umap.UMAP(n_components=2, random_state=cfg["seed"]).fit_transform(x)
            return reduced, coords
        except Exception as exc:  # umap optional / may be unavailable on py3.13
            print(f"[cluster] UMAP unavailable ({exc}); using 2-D PCA for plotting.")
    coords = PCA(n_components=2, random_state=cfg["seed"]).fit_transform(x)
    return reduced, coords


def run_clustering(cfg: Config) -> Path:
    """Cluster embeddings and write per-object cluster labels + 2-D coords."""
    x = np.load(cfg.outputs_dir / "embeddings.npy")
    index = pd.read_csv(cfg.outputs_dir / "embeddings_index.csv")
    c = cfg["cluster"]

    if c["l2_normalize"]:
        x = normalize(x)

    features, coords = _reduce(x, cfg)

    if c["algorithm"] == "hdbscan":
        h = c["hdbscan"]
        labels = HDBSCAN(
            min_cluster_size=int(h["min_cluster_size"]),
            min_samples=(None if h["min_samples"] is None else int(h["min_samples"])),
            metric=h["metric"],
        ).fit_predict(features)
    elif c["algorithm"] == "kmeans":
        k = int(c["kmeans"]["n_clusters"])
        labels = KMeans(n_clusters=k, random_state=cfg["seed"], n_init="auto").fit_predict(features)
    else:
        raise ValueError(f"Unknown cluster.algorithm: {c['algorithm']!r}")

    out_df = index.copy()
    out_df["cluster"] = labels
    out_df["x"] = coords[:, 0]
    out_df["y"] = coords[:, 1]

    out = cfg.outputs_dir / "clusters.csv"
    out_df.to_csv(out, index=False)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"Found {n_clusters} morphotype clusters "
          f"({n_noise}/{len(labels)} objects in noise bucket -1) -> {out}")
    return out
