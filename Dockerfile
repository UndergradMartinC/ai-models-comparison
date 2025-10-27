# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, PyTorch, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install PyTorch and dependencies (CPU-only for smaller image size)
# For GPU support, use: torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir \
    transformers>=4.30.0 \
    opencv-python-headless>=4.8.0 \
    pillow>=9.5.0 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    fastapi \
    uvicorn[standard] \
    python-multipart

# Install GroundingDINO from GitHub
RUN pip install --no-cache-dir git+https://github.com/IDEA-Research/GroundingDINO.git

# Copy only necessary application files
COPY dinoAPI.py .
COPY grounding_dino.py .
COPY model_tests.py .
COPY COCO_CLASSES.py .

# Copy model weights and configs
COPY weights/ ./weights/

# Create directories for uploads and outputs
RUN mkdir -p uploads outputs

# Expose port 8080
EXPOSE 8080

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Healthcheck to ensure the API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/docs').getcode()" || exit 1

# Run the API
CMD ["python", "dinoAPI.py"]

