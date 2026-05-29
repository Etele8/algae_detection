"""Configuration loading and path helpers.

A single ``config.yaml`` at the project root drives every stage. Loading it
returns a nested ``dict`` plus resolved absolute paths, so scripts never have
to guess where the project root is.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Parsed configuration with resolved absolute paths."""

    raw: dict[str, Any]
    root: Path

    @property
    def data_dir(self) -> Path:
        return self._resolve(self.raw["paths"]["data_dir"])

    @property
    def outputs_dir(self) -> Path:
        return self._resolve(self.raw["paths"]["outputs_dir"])

    @property
    def models_dir(self) -> Path:
        # An env override lets a prebuilt Docker image ship weights in a fixed
        # location (e.g. /opt/models) that a mounted /workspace volume can't
        # shadow. Falls back to the configured path.
        env = os.environ.get("ALGAE_MODELS_DIR")
        if env:
            return Path(env)
        return self._resolve(self.raw["paths"]["models_dir"])

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (self.root / path)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def resolve_device(value: str | None) -> str:
    """Resolve a config device string to a concrete torch device.

    ``"auto"`` (or ``None``) -> ``"cuda"`` when a GPU is visible, else ``"cpu"``.
    This lets the identical codebase run on the CPU dev box and on a RunPod GPU
    without editing configs.
    """
    if value and value != "auto":
        return value
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_config(path: str | Path | None = None) -> Config:
    """Load ``config.yaml`` (defaults to the one at the project root)."""
    root = project_root()
    cfg_path = Path(path) if path else (root / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw=raw, root=root)
    # Ensure output dirs exist so downstream stages can just write.
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    return cfg
