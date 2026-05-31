"""Stage 4 - Clustering: group organism embeddings into morphotypes.

DINOv2 embeddings are high-dimensional (384-1536d). Running a density-based
clusterer directly on them - or on a plain PCA of them - tends to collapse
everything into one "noise" blob, because distances concentrate in high
dimensions. The robust, well-established recipe is to **reduce with UMAP first
and cluster in that low-dimensional space**, where HDBSCAN finds clusters of
varying density and routes rare/odd organisms to a noise bucket (label ``-1``).
A separate 2-D UMAP is saved for the review scatter plot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .config import Config


def _features_and_coords(x: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (features_for_clustering, coords_2d_for_plotting).

    With ``reducer: umap`` we cluster in a moderate-dim UMAP space (not 2-D,
    which over-compresses) and plot in a separate 2-D UMAP. ``min_dist: 0`` is
    the recommended setting when the UMAP output feeds a clusterer.
    """
    c = cfg["cluster"]
    seed = cfg["seed"]

    if c["reducer"] == "umap":
        try:
            import umap

            u = c.get("umap") or {}
            n_components = int(u.get("n_components", 10))
            n_neighbors = int(u.get("n_neighbors", 15))
            min_dist = float(u.get("min_dist", 0.0))
            metric = u.get("metric", "cosine")
            feats = umap.UMAP(
                n_components=n_components, n_neighbors=n_neighbors,
                min_dist=min_dist, metric=metric, random_state=seed,
            ).fit_transform(x)
            coords = umap.UMAP(
                n_components=2, n_neighbors=n_neighbors,
                min_dist=min_dist, metric=metric, random_state=seed,
            ).fit_transform(x)
            return feats, coords
        except Exception as exc:  # umap optional / may be unavailable
            print(f"[cluster] UMAP unavailable ({exc}); falling back to PCA.")

    n_comp = min(int(c["pca_components"]), x.shape[0], x.shape[1])
    feats = PCA(n_components=n_comp, random_state=seed).fit_transform(x)
    coords = PCA(n_components=2, random_state=seed).fit_transform(x)
    return feats, coords


def run_clustering(cfg: Config) -> Path:
    """Cluster embeddings and write per-object cluster labels + 2-D coords."""
    x = np.load(cfg.outputs_dir / "embeddings.npy")
    index = pd.read_csv(cfg.outputs_dir / "embeddings_index.csv")
    c = cfg["cluster"]

    # Optional focus prune: exclude blurry/defocused crops from clustering
    # without re-detecting/embedding (uses the focus column in objects.csv).
    min_focus = float(c.get("min_focus") or 0.0)
    if min_focus > 0:
        obj = pd.read_csv(cfg.outputs_dir / "objects.csv")
        if "focus" in obj.columns:
            keep_ids = set(obj.loc[obj["focus"] >= min_focus, "object_id"])
            keep = index["object_id"].isin(keep_ids).to_numpy()
            print(f"[cluster] focus >= {min_focus}: kept {int(keep.sum())}, "
                  f"dropped {int((~keep).sum())} blurry of {len(index)}")
            x = x[keep]
            index = index[keep].reset_index(drop=True)
        else:
            print("[cluster] cluster.min_focus set but objects.csv has no "
                  "'focus' column (re-run detect to add it); skipping prune")

    if c["l2_normalize"]:
        x = normalize(x)

    features, coords = _features_and_coords(x, cfg)

    if c["algorithm"] == "hdbscan":
        h = c["hdbscan"]
        labels = HDBSCAN(
            min_cluster_size=int(h["min_cluster_size"]),
            min_samples=(None if h["min_samples"] is None else int(h["min_samples"])),
            metric=h["metric"],
            # 'leaf' selects the fine leaf nodes of the hierarchy -> more, smaller,
            # more homogeneous clusters (better separation than the default 'eom').
            cluster_selection_method=h.get("cluster_selection_method", "eom"),
            # >0 merges clusters whose distance is below this; 0 keeps them split.
            cluster_selection_epsilon=float(h.get("cluster_selection_epsilon", 0.0)),
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
