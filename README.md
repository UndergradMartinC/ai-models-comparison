# AI Models Comparison

Real object detection system comparing YOLOX and Grounding DINO models.

## 🚀 Quick Start

### Download Model Weights

Before running, download the required model weights:

```bash
# Create weights directory
mkdir -p weights

# Download YOLOX-X weights (757MB)
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth

# Download Grounding DINO weights (662MB)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights/

# Download Grounding DINO config
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py -P weights/
```

## 🔍 Run Individual Models

### YOLOX Standalone
```bash
python3 yolox_detector.py
```
- Detects objects using COCO classes (80 categories)
- Outputs: `outputs/yolox_*_detections.json` and `outputs/yolox_*_visualization.jpg`

### Grounding DINO Standalone  
```bash
python3 grounding_dino.py
```
- Open vocabulary detection (75+ object types)
- Outputs: `outputs/dino_*_detections.json` and `outputs/dino_*_visualization.jpg`

## ⏱️ Timing Comparison

Compare all models with timing analysis:
```bash
python3 main.py
```

## 📊 Features

- **Real Detection**: Uses actual pretrained weights, not fake results
- **Accurate Bounding Boxes**: Fixed coordinate conversion issues
- **Comprehensive Analysis**: Category breakdown, confidence statistics
- **Visual Output**: Color-coded detection visualizations
- **JSON Export**: Structured detection data
