#!/usr/bin/env python
"""Thin launcher so the pipeline runs without installing the package.

Adds ``src/`` to the path and dispatches to ``algae.cli``.
Example: ``python run.py unsupervised``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from algae.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
