# Batteries-included image for RunPod GPU pods.
# CUDA + PyTorch + all Python deps + the SAM2.1-large and DINOv2-giant weights
# are baked in, so an activated pod is ready to run with NO setup step.
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

# System deps. ultralytics force-installs the GUI build of opencv, which links
# libGL/libxcb at import time, so we install those runtime libs even though we
# also keep the headless wheel. git is for cloning code on the pod.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ca-certificates \
        libglib2.0-0 libgl1 libxcb1 libsm6 libxext6 libxrender1 \
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
#    no per-pod download). Strongest models: SAM2.1-large + DINOv2-giant.
#    Names must match config.gpu.yaml; change both together if you switch.
WORKDIR /opt/models
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
AutoImageProcessor.from_pretrained('facebook/dinov2-giant'); \
AutoModel.from_pretrained('facebook/dinov2-giant')" \
 && python -c "from ultralytics import SAM; SAM('sam2.1_l.pt')"

# 4) Project code (data/outputs/weights excluded via .dockerignore).
WORKDIR /opt/algae_detection
COPY . /opt/algae_detection

# Sanity check at build time: GPU code path imports cleanly.
RUN python -c "import torch, ultralytics, transformers, cv2, sklearn; print('image deps OK')"

# 5) RunPod entrypoint: keep the container alive (so the pod stays up) and
#    enable key-based SSH. Kept as a late layer so rebuilds reuse the cached
#    torch / deps / weights layers above and only push small new layers.
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/*
COPY start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh   # strip CR in case of CRLF
CMD ["/start.sh"]
