"""Stage 6 - Classify: turn named clusters into a reusable classifier.

Once the researcher writes a ``label`` for each cluster in
``cluster_labels.csv``, every object inherits its cluster's label. We then fit a
k-NN classifier in the DINOv2 embedding space. That classifier is the payoff:
on the *full* dataset, new organisms are embedded and assigned a taxon
automatically (with a confidence = neighbour vote fraction), so the researcher
only ever curates groups, never individual cells.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

_MODEL_FILE = "classifier.joblib"


def _load_labeled(cfg: Config):
    """Return (embeddings, object_ids, labels) for objects in named clusters."""
    x = np.load(cfg.outputs_dir / "embeddings.npy")
    index = pd.read_csv(cfg.outputs_dir / "embeddings_index.csv")
    clusters = pd.read_csv(cfg.outputs_dir / "clusters.csv")
    labels_path = cfg.outputs_dir / "cluster_labels.csv"
    if not labels_path.exists():
        raise RuntimeError(
            "cluster_labels.csv not found - run the review stage and fill in a "
            "label for each cluster first."
        )
    label_map = pd.read_csv(labels_path)
    label_map["label"] = label_map["label"].fillna("").astype(str).str.strip()
    named = label_map[label_map["label"] != ""]
    if named.empty:
        raise RuntimeError(
            "No labels filled in yet. Edit outputs/cluster_labels.csv and give "
            "each meaningful cluster a `label`."
        )

    cluster_to_label = dict(zip(named["cluster"], named["label"]))
    merged = index.merge(clusters[["object_id", "cluster"]], on="object_id", how="left")
    merged["label"] = merged["cluster"].map(cluster_to_label)
    mask = merged["label"].notna().to_numpy()
    return x[mask], merged.loc[mask, "object_id"].to_numpy(), merged.loc[mask, "label"].to_numpy()


def train_classifier(cfg: Config) -> Path:
    """Fit and persist a k-NN classifier over labeled object embeddings."""
    import joblib
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    x, _, y = _load_labeled(cfg)
    k = min(int(cfg["classify"]["knn_k"]), len(x))
    clf = make_pipeline(
        Normalizer(),                       # cosine-like geometry, matches clustering
        KNeighborsClassifier(n_neighbors=k, weights="distance"),
    )
    clf.fit(x, y)

    out = cfg.outputs_dir / _MODEL_FILE
    joblib.dump({"model": clf, "classes": sorted(set(y))}, out)
    print(f"Trained k-NN on {len(x)} labeled objects across "
          f"{len(set(y))} classes (k={k}) -> {out}")
    return out


def predict(cfg: Config, embeddings: np.ndarray) -> pd.DataFrame:
    """Predict labels + confidence for an (N, D) embedding matrix."""
    import joblib

    bundle = joblib.load(cfg.outputs_dir / _MODEL_FILE)
    clf = bundle["model"]
    proba = clf.predict_proba(embeddings)
    classes = clf.classes_
    pred_idx = proba.argmax(axis=1)
    return pd.DataFrame(
        {
            "label": classes[pred_idx],
            "confidence": proba[np.arange(len(proba)), pred_idx],
        }
    )


def label_all_objects(cfg: Config) -> Path:
    """Apply the trained classifier to every detected object and save results.

    This produces the deliverable table: each detected organism with its
    predicted taxon + confidence, ready for the researcher to review or export.
    """
    x = np.load(cfg.outputs_dir / "embeddings.npy")
    index = pd.read_csv(cfg.outputs_dir / "embeddings_index.csv")
    preds = predict(cfg, x)
    result = pd.concat([index.reset_index(drop=True), preds], axis=1)

    objects = pd.read_csv(cfg.outputs_dir / "objects.csv")
    result = result.merge(
        objects[["object_id", "image_id", "raw_path"]], on="object_id", how="left"
    )
    out = cfg.outputs_dir / "predictions.csv"
    result.to_csv(out, index=False)
    print(f"Wrote predictions for {len(result)} objects -> {out}")
    return out
