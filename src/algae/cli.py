"""Command-line entry point for the pipeline.

Usage (from the project root, with the venv active)::

    python run.py manifest        # 1. scan data, drop annotated _m twins
    python run.py detect          # 2. segment organisms -> crops + objects.csv
    python run.py embed           # 3. DINOv2 features -> embeddings.npy
    python run.py cluster         # 4. HDBSCAN morphotypes -> clusters.csv
    python run.py review          # 5. contact sheets + cluster_labels.csv
    # ... researcher fills outputs/cluster_labels.csv ...
    python run.py train           # 6. fit k-NN over labeled clusters
    python run.py predict         # 7. label every object -> predictions.csv

    python run.py unsupervised    # runs steps 1-5 in one go
"""
from __future__ import annotations

import argparse

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="algae", description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["manifest", "detect", "embed", "cluster", "review", "visualize",
                 "train", "predict", "unsupervised", "all"]:
        sub.add_parser(name)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    # Imported lazily so cheap commands don't pay torch/ultralytics import cost.
    if args.cmd == "manifest":
        from .manifest import write_manifest
        print("manifest ->", write_manifest(cfg))
    elif args.cmd == "detect":
        from .detect import run_detection
        run_detection(cfg)
    elif args.cmd == "embed":
        from .embed import run_embedding
        run_embedding(cfg)
    elif args.cmd == "cluster":
        from .cluster import run_clustering
        run_clustering(cfg)
    elif args.cmd == "review":
        from .review import run_review
        run_review(cfg)
    elif args.cmd == "visualize":
        from .visualize import run_visualize
        run_visualize(cfg)
    elif args.cmd == "train":
        from .classify import train_classifier
        train_classifier(cfg)
    elif args.cmd == "predict":
        from .classify import label_all_objects
        label_all_objects(cfg)
    elif args.cmd in ("unsupervised", "all"):
        from .manifest import write_manifest
        from .detect import run_detection
        from .embed import run_embedding
        from .cluster import run_clustering
        from .review import run_review
        from .visualize import run_visualize
        write_manifest(cfg)
        run_detection(cfg)
        run_embedding(cfg)
        run_clustering(cfg)
        run_review(cfg)
        run_visualize(cfg)
        if args.cmd == "all":
            from .classify import train_classifier, label_all_objects
            train_classifier(cfg)
            label_all_objects(cfg)


if __name__ == "__main__":
    main()
