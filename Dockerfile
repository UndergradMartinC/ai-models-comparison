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

# GroundingDINO: clone at pinned commit, install package
ARG GD_REF=856dde20aee659246248e20734ef9ba5214f5e44
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /opt/GroundingDINO && \
    cd /opt/GroundingDINO && git checkout ${GD_REF}
RUN python3.11 -m pip install --no-build-isolation -v /opt/GroundingDINO

# Build CUDA ops from the cloned repo and verify _C exists
RUN python3.11 - <<'PY'
import importlib, os, pathlib, subprocess, sys

root = pathlib.Path('/opt/GroundingDINO')
print('Scanning for GroundingDINO ops under:', root)

def listdir_safe(p: pathlib.Path):
    try:
        return os.listdir(p)
    except Exception:
        return []

candidates = []
for dirpath, dirnames, filenames in os.walk(root):
    dirpath_p = pathlib.Path(dirpath)
    if 'setup.py' in filenames:
        names = set(filenames)
        has_ms_file = any('ms_deform_attn' in f for f in names)
        src_dir_matches = []
        for sub in ('src', 'csrc', 'CUDA', 'cuda'):
            subp = dirpath_p / sub
            if subp.is_dir():
                if any('ms_deform_attn' in f for f in listdir_safe(subp)):
                    src_dir_matches.append(sub)
        if has_ms_file or src_dir_matches or 'ops' in dirpath:
            candidates.append(dirpath_p)

if not candidates:
    # Fallback: find directories containing ms_deform_attn*.py and look for setup.py alongside
    for dirpath, dirnames, filenames in os.walk(root):
        if any('ms_deform_attn' in f for f in filenames):
            p = pathlib.Path(dirpath)
            if (p / 'setup.py').exists():
                candidates.append(p)

if not candidates:
    raise FileNotFoundError('Could not find GroundingDINO ops setup.py in repo')

print('Found candidate ops dirs:', candidates)
subprocess.check_call([sys.executable, 'setup.py', 'build', 'install'], cwd=str(candidates[0]))

m = importlib.import_module('groundingdino.models.GroundingDINO.ms_deform_attn')
assert hasattr(m, '_C'), 'GroundingDINO ops _C not built!'
print('Built:', m.__file__)
PY

# Stash the entire package (code + compiled .so)
RUN python3.11 - <<'PY'
import importlib, pathlib, os, shutil
pkg = importlib.import_module("groundingdino")
pkg_dir = pathlib.Path(pkg.__file__).parent
artifacts = pathlib.Path("/artifacts")
artifacts.mkdir(parents=True, exist_ok=True)
dest = artifacts / "groundingdino"
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(pkg_dir, dest)
print("Copied GroundingDINO from:", pkg_dir)
PY


# =========================
# STAGE 2: RUNTIME (SLIM)
# =========================
FROM python:3.11-slim
WORKDIR /app

# Minimal runtime libs for OpenCV, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA-enabled Torch wheels (bundle CUDA/cuDNN)
RUN pip install -U pip setuptools wheel && \
    pip install --no-cache-dir \
      "transformers>=4.30.0" \
      "opencv-python-headless>=4.8.0" \
      "pillow>=9.5.0" \
      "numpy>=1.24.0" \
      "matplotlib>=3.7.0" \
      fastapi uvicorn[standard] python-multipart

# Copy the prebuilt GroundingDINO package from builder (no pip install here)
COPY --from=builder /artifacts/groundingdino /usr/local/lib/python3.11/site-packages/groundingdino

# Optional: extra deps from your requirements.txt
COPY requirements.txt .
RUN [ -s requirements.txt ] && pip install --no-cache-dir -r requirements.txt || true

# App files
COPY dinoAPI.py grounding_dino.py model_tests.py COCO_CLASSES.py ./
COPY weights/ ./weights/
RUN mkdir -p uploads outputs

# Sanity check (fail fast if _C missing)
RUN python - <<'PY'
import importlib, torch
m = importlib.import_module("groundingdino.models.GroundingDINO.ms_deform_attn")
print("Torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available())
print("GroundingDINO _C present:", hasattr(m, "_C"), "path:", m.__file__)
PY


EXPOSE 8080
ENV HOST=0.0.0.0 PORT=8080 PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/docs').getcode()" || exit 1

CMD ["python", "dinoAPI.py"]
