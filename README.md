# Algae detection — unsupervised morphotype clustering

Microscopy phytoplankton analysis without per-cell labels. The model finds
every organism in each frame and groups look-alikes into **morphotype
clusters**; a researcher then names whole **groups** instead of inspecting and
classifying each organism by hand. Those group names train a classifier that
scales to the full dataset.

Data: brightfield microscopy frames (40× objective) of water samples, e.g.
`Deseda_Kaposvariveg_20250713_300_20_40x*.png`. Files ending in `_m` are the
microscope's annotated copies (yellow measurement overlays) and are ignored by
default — see [config.yaml](config.yaml) → `data.drop_annotated_suffix`.

## Pipeline

| Stage | Module | What it does | Output |
|------:|--------|--------------|--------|
| 1 | `manifest` | Scan `data/`, drop `_m` annotated twins | `outputs/manifest.csv` |
| 2 | `detect`   | "Segment everything" (MobileSAM/FastSAM), filter to organism-sized masks | `outputs/objects.csv`, `outputs/crops/` |
| 3 | `embed`    | DINOv2 ViT features per crop (background-masked, aspect-preserving) | `outputs/embeddings.npy` |
| 4 | `cluster`  | PCA → HDBSCAN morphotypes (+ 2-D projection) | `outputs/clusters.csv` |
| 5 | `review`   | Contact sheet per cluster + scatter + labeling template | `outputs/clusters/`, `outputs/cluster_labels.csv` |
| 5b | `visualize` | Overlay masks + cluster/label onto each full frame (QA) | `outputs/overlays/` |
| 6 | `train`    | Fit k-NN over the researcher's named clusters | `outputs/classifier.joblib` |
| 7 | `predict`  | Label every detected organism + confidence | `outputs/predictions.csv` |

**Why these choices** (no labels, accuracy-first):
- *Detection* uses a class-agnostic foundation segmenter — no training data
  needed, and it generalizes across taxa. A `classical` OpenCV backend is the
  fallback.
- *Embeddings* use **DINOv2**, self-supervised features that already separate
  shapes/textures, so clustering is meaningful with zero fine-tuning.
- *Clustering* uses **HDBSCAN**: it discovers the number of groups itself and
  routes rare/ambiguous organisms to a noise bucket (`-1`) for manual review.

### CPU (local) vs GPU (RunPod)

The code is device-agnostic (`device: auto` → CUDA if present, else CPU). Two
profiles ship in the repo:

| | Local CPU ([config.yaml](config.yaml)) | GPU / RunPod ([config.gpu.yaml](config.gpu.yaml)) |
|--|--|--|
| Detection | MobileSAM | **SAM2.1-large** |
| Embedding | DINOv2-small (384-d) | **DINOv2-large** (1024-d) |
| Speed | ~130 s/frame (segmentation) | seconds/frame |
| Use for | code checks, small subsets | full, high-accuracy runs |

MobileSAM on CPU measured **~130 s per frame**, so CPU is only practical for
correctness checks (use `detect.max_images: 3`). Real runs and the larger
models belong on a GPU.

## Setup

```powershell
# venv already at .venv (Python 3.13)
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
# Unsupervised part end-to-end (stages 1–5)
.venv\Scripts\python.exe run.py unsupervised

# 1. open outputs/clusters/*.png, then fill the `label` column in
#    outputs/cluster_labels.csv (one taxon name per cluster)

# Train + apply the classifier (stages 6–7)
.venv\Scripts\python.exe run.py train
.venv\Scripts\python.exe run.py predict
```

Each stage can also be run individually (`run.py detect`, `run.py cluster`, …).
All parameters live in [config.yaml](config.yaml).

## Running on GPU (RunPod)

For real, high-quality runs use a GPU pod. The same code runs unchanged — only
the config profile differs.

1. **Create a pod** with a PyTorch GPU template (CUDA torch preinstalled). A
   16–24 GB GPU (e.g. RTX A4000/4090) comfortably fits SAM2-large + DINOv2-large.
2. **Get the project + data onto the pod** (pick one):
   ```bash
   git clone <your-repo-url> /workspace/algae_detection      # code
   # data/ is gitignored — copy it separately, e.g.:
   runpodctl receive <code>            # or scp / rsync / a mounted volume
   ```
3. **Install deps** (installs CUDA torch only if missing):
   ```bash
   cd /workspace/algae_detection
   bash setup_runpod.sh
   ```
4. **Run the high-quality pipeline:**
   ```bash
   python run.py --config config.gpu.yaml unsupervised   # stages 1–5
   # download outputs/ to your machine, review contact sheets, fill labels
   python run.py --config config.gpu.yaml train
   python run.py --config config.gpu.yaml predict
   ```
5. **Pull results back** (`outputs/` holds crops, clusters, contact sheets,
   `predictions.csv`) before stopping the pod.

`config.gpu.yaml` selects SAM2.1-large, DINOv2-large, UMAP projection and
larger batches. Lower `detect.weights` to `sam2.1_b.pt` or `embed.model_name`
to `facebook/dinov2-base` if VRAM is tight; raise to `dinov2-giant` for maximum
accuracy on a big GPU.

### Custom RunPod image (no per-pod setup)

`bash setup_runpod.sh` works but reinstalls deps + re-downloads weights every
time a fresh pod starts. To skip that entirely, build the [Dockerfile](Dockerfile)
once — it bakes CUDA + torch + all deps **and** the SAM2-large / DINOv2-large
weights into the image (under `/opt`, which RunPod's `/workspace` volume can't
shadow). Then an activated pod is instantly ready.

**Build & push once** (needs Docker + a registry login, e.g. Docker Hub):

```bash
docker build -t <user>/algae:gpu .
docker push  <user>/algae:gpu
# ~10 GB image (CUDA + torch + weights); building pulls the weights from the net.
```

**Use it on RunPod:** create a pod / template with **Container Image** =
`<user>/algae:gpu`. On the pod, no install is needed — just get your code/data
in place and run:

```bash
cd /opt/algae_detection           # code is baked in here
ln -s /workspace/data data        # or copy your frames into ./data
python run.py --config config.gpu.yaml unsupervised
```

Weights load from the image cache (`ALGAE_MODELS_DIR=/opt/models`, `HF_HOME`),
so there are **no downloads at runtime**. If you change the models in
`config.gpu.yaml`, update the pre-download names in the Dockerfile and rebuild
(or accept a one-time download on first use).

> Building a ~10 GB image on Windows is heavy but one-time. Alternatively build
> it in CI (GitHub Actions) and push to your registry.

## Scaling to the full dataset

The 72 frames here are a sample of a larger archive. To process more data:
drop new frames into `data/` (or point `paths.data_dir` elsewhere) and re-run
`detect → embed → predict`. New organisms are assigned the nearest named
morphotype with a confidence score; low-confidence objects are the ones worth a
human glance. Re-run `cluster → review → train` periodically as new
morphotypes appear.
