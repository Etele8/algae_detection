# Batteries-included image for RunPod GPU pods.
# CUDA + PyTorch + all Python deps + the SAM2-large and DINOv2-large weights are
# baked in, so an activated pod is ready to run with NO setup step.
#
# Build & push (see README "Custom RunPod image"):
#     docker build -t <user>/algae:gpu .
#     docker push  <user>/algae:gpu
# Then point a RunPod template at <user>/algae:gpu.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model caches live under /opt, NOT /workspace — RunPod mounts a volume at
    # /workspace which would otherwise hide anything baked there.
    ALGAE_MODELS_DIR=/opt/models \
    HF_HOME=/opt/models/hf \
    TORCH_HOME=/opt/models/torch \
    YOLO_CONFIG_DIR=/opt/models/ultralytics

# System deps. opencv-python-headless needs libglib2.0-0; git for cloning code.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ca-certificates libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# 1) CUDA build of torch (own layer so it caches independently of app deps).
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu124
RUN pip install torch torchvision --index-url ${CUDA_INDEX}

# 2) The remaining Python dependencies.
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install -r /tmp/requirements-gpu.txt

# 3) Pre-download model weights INTO THE IMAGE (this is the whole point —
#    no per-pod download). Names match config.gpu.yaml; change here + there
#    together if you switch models (e.g. dinov2-giant / sam2.1_b.pt).
WORKDIR /opt/models
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
AutoImageProcessor.from_pretrained('facebook/dinov2-large'); \
AutoModel.from_pretrained('facebook/dinov2-large')" \
 && python -c "from ultralytics import SAM; SAM('sam2.1_l.pt')"

# 4) Project code (data/outputs/weights excluded via .dockerignore).
WORKDIR /opt/algae_detection
COPY . /opt/algae_detection

# Sanity check at build time: GPU code path imports cleanly.
RUN python -c "import torch, ultralytics, transformers, cv2, sklearn; print('image deps OK')"

CMD ["bash"]
