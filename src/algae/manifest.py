"""Build a manifest of input images.

The microscope exports an annotated copy of some frames with a ``_m`` suffix
(yellow measurement overlays burned into the pixels). Those are near-duplicates
of a clean frame and would (a) pollute clustering with a fake "has yellow text"
morphotype and (b) double-count organisms. By default we keep the clean frame
and record the annotated twin for reference only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from .config import Config


def _strip_suffix(stem: str, suffix: str) -> str:
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def build_manifest(cfg: Config) -> pd.DataFrame:
    """Scan ``data_dir`` and return one row per image with annotation flags."""
    glob = cfg["data"]["image_glob"]
    suffix = cfg["data"]["drop_annotated_suffix"]

    rows = []
    for path in sorted(cfg.data_dir.glob(glob)):
        stem = path.stem
        is_annotated = stem.endswith(suffix)
        clean_stem = _strip_suffix(stem, suffix)
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
        rows.append(
            {
                "image_id": stem,
                "path": str(path),
                "filename": path.name,
                "is_annotated": is_annotated,
                "clean_id": clean_stem,
                "width": w,
                "height": h,
                "mode": mode,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images matching {glob!r} found in {cfg.data_dir}")

    # A clean frame is "used" for the pipeline; annotated twins are kept for
    # reference but excluded from detection/embedding/clustering.
    clean_ids = set(df.loc[~df["is_annotated"], "clean_id"])
    df["use"] = ~df["is_annotated"]
    df["has_clean_twin"] = df["clean_id"].isin(clean_ids)
    return df


def write_manifest(cfg: Config) -> Path:
    """Build the manifest and persist it to ``outputs/manifest.csv``."""
    df = build_manifest(cfg)
    out = cfg.outputs_dir / "manifest.csv"
    df.to_csv(out, index=False)
    return out


def load_manifest(cfg: Config) -> pd.DataFrame:
    """Load the manifest, building it on demand if missing."""
    out = cfg.outputs_dir / "manifest.csv"
    if not out.exists():
        write_manifest(cfg)
    return pd.read_csv(out)


def usable_images(cfg: Config) -> pd.DataFrame:
    """Return only the clean frames used by the pipeline (``use == True``)."""
    df = load_manifest(cfg)
    return df[df["use"]].reset_index(drop=True)
