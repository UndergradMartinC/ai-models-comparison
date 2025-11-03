# Single-stage: build + run in one image (most reliable)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps: Python 3.11 + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    git build-essential ninja-build curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# PyTorch (CUDA 12.1 wheels) — versions known-good with many DINO stacks
RUN python3.11 -m pip install -U pip setuptools wheel && \
    python3.11 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.4.1 torchvision==0.19.1

# App deps
RUN python3.11 -m pip install --no-cache-dir \
      "transformers>=4.30.0" \
      "opencv-python-headless>=4.8.0" \
      "pillow>=9.5.0" \
      "numpy>=1.24.0" \
      "matplotlib>=3.7.0" \
      fastapi uvicorn[standard] python-multipart

# GroundingDINO from source (no build isolation so it sees torch/cuda)
# Pin to a stable commit; update if you need a newer API
ARG GD_REF=856dde20aee659246248e20734ef9ba5214f5e44
RUN python3.11 -m pip install --no-build-isolation \
    "git+https://github.com/IDEA-Research/GroundingDINO.git@${GD_REF}#egg=groundingdino"

# --- Your app files ---
COPY dinoAPI.py grounding_dino.py model_tests.py COCO_CLASSES.py ./
COPY weights/ ./weights/
RUN mkdir -p uploads outputs

# Sanity check: verify CUDA, Torch, and the _C extension
RUN python3.11 - <<'PY'
import importlib, torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "CUDA avail:", torch.cuda.is_available())
m = importlib.import_module("groundingdino.models.GroundingDINO.ms_deform_attn")
print("GroundingDINO _C present:", hasattr(m, "_C"), "at:", m.__file__)
PY

EXPOSE 8080
ENV HOST=0.0.0.0 PORT=8080 PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/docs').getcode()" || exit 1

CMD ["python3.11", "dinoAPI.py"]
