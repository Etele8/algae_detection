#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment setup for a RunPod (or any CUDA) GPU instance.
#
# Recommended pod template: a PyTorch image (e.g. "RunPod PyTorch 2.x") so a
# CUDA-enabled torch is already present. This script installs torch only if it
# is missing or CPU-only, then the rest of the deps.
#
# Usage on the pod:
#     cd /workspace/algae_detection
#     bash setup_runpod.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PY=${PYTHON:-python}
CUDA_INDEX=${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}

echo ">> Python: $($PY --version)"

if $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo ">> CUDA torch already present: $($PY -c 'import torch; print(torch.__version__)')"
else
    echo ">> Installing CUDA torch/torchvision from $CUDA_INDEX"
    $PY -m pip install --upgrade pip
    $PY -m pip install torch torchvision --index-url "$CUDA_INDEX"
fi

echo ">> Installing project dependencies (requirements-gpu.txt)"
$PY -m pip install -r requirements-gpu.txt

echo ">> Verifying GPU is visible to torch:"
$PY - <<'PYCODE'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PYCODE

echo ">> Done. Run the pipeline with:"
echo "   $PY run.py --config config.gpu.yaml unsupervised"
