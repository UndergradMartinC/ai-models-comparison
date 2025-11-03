## =========================
## STAGE 1: BUILD EXTENSIONS
## =========================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder
WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive
# Python + build deps (builder only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install -U pip setuptools wheel

# Match Torch to CUDA 12.1 (important!)
RUN python3.11 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1

# GroundingDINO (pin the commit; NO build isolation so it sees torch)
ARG GD_REF=856dde20aee659246248e20734ef9ba5214f5e44
RUN python3.11 -m pip install --no-build-isolation \
  "git+https://github.com/IDEA-Research/GroundingDINO.git@${GD_REF}#egg=groundingdino"

## Force-build CUDA ops and verify _C exists
#RUN python3.11 - <<'PY'
#import importlib, pathlib, subprocess, sys
#pkg = importlib.import_module("groundingdino")
#ops = pathlib.Path(pkg.__file__).parent / "models" / "ops"
#subprocess.check_call([sys.executable, "setup.py", "build", "install"], cwd=str(ops))
#m = importlib.import_module("groundingdino.models.GroundingDINO.ms_deform_attn")
#assert hasattr(m, "_C"), "GroundingDINO ops _C not built!"
#print("Built:", m.__file__)
#PY
#
## Stash the entire package (code + compiled .so)
#RUN mkdir -p /artifacts && \
#    cp -r /usr/local/lib/python3.11/site-packages/groundingdino /artifacts/groundingdino
#
#
## =========================
## STAGE 2: RUNTIME (SLIM)
## =========================
#FROM python:3.11-slim
#WORKDIR /app
#
## Minimal runtime libs for OpenCV, etc.
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 \
#    && rm -rf /var/lib/apt/lists/*
#
## CUDA-enabled Torch wheels (bundle CUDA/cuDNN)
#RUN pip install -U pip setuptools wheel && \
#    pip install --no-cache-dir \
#      "transformers>=4.30.0" \
#      "opencv-python-headless>=4.8.0" \
#      "pillow>=9.5.0" \
#      "numpy>=1.24.0" \
#      "matplotlib>=3.7.0" \
#      fastapi uvicorn[standard] python-multipart
#
## Copy the prebuilt GroundingDINO package from builder (no pip install here)
#COPY --from=builder /artifacts/groundingdino /usr/local/lib/python3.11/site-packages/groundingdino
#
## Optional: extra deps from your requirements.txt
#COPY requirements.txt .
#RUN [ -s requirements.txt ] && pip install --no-cache-dir -r requirements.txt || true
#
## App files
#COPY dinoAPI.py grounding_dino.py model_tests.py COCO_CLASSES.py ./
#COPY weights/ ./weights/
#RUN mkdir -p uploads outputs
#
## Sanity check (fail fast if _C missing)
#RUN python - <<'PY'
#import importlib, torch
#m = importlib.import_module("groundingdino.models.GroundingDINO.ms_deform_attn")
#print("Torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available())
#print("GroundingDINO _C present:", hasattr(m, "_C"), "path:", m.__file__)
#PY
#
#
#EXPOSE 8080
#ENV HOST=0.0.0.0 PORT=8080 PYTHONUNBUFFERED=1
#
#HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/docs').getcode()" || exit 1
#
#CMD ["python", "dinoAPI.py"]
#