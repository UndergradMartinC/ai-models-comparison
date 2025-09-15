# ai-models-comparison

Simple harness to compare models and a standalone YOLOX demo.

## Run comparison

```bash
python3 main.py
```

## YOLOX standalone demo

Run a stubbed standalone YOLOX demo that simulates inference and prints results.

```bash
python3 yolox_demo.py --image /path/to/image.jpg --num 5 --seed 42 --json --save ./outputs/yolox_demo.json
```

Arguments:
- `--image`: Optional path to an image. If provided, existence is validated.
- `--num`: Number of fake detections to generate (default 3).
- `--seed`: Optional seed for deterministic demo outputs.
- `--json`: Print results as JSON (otherwise human-readable text).
- `--save`: Optional path to save JSON results to disk.

### Two-image comparison mode

Compare two images of the same room and see detected differences using IoU matching:

```bash
python3 yolox_demo.py --image-a /path/room_before.jpg --image-b /path/room_after.jpg --num 8 --iou 0.5 --json --save ./outputs/yolox_compare.json --save-viz ./outputs/yolox_compare_viz.jpg
```

Arguments:
- `--image-a` / `--image-b`: Paths to the two images.
- `--iou`: IoU threshold for matching boxes (default 0.5).
- `--save-viz`: Save a final annotated image (green=added in B, red=removed from A, yellow=class changed among matches). Requires Pillow (`pip install pillow`).

### Pixel-diff comparison (less random, more deterministic)

Use a simple pixel-difference heuristic instead of random detections:

```bash
python3 yolox_demo.py --image-a before.jpg --image-b after.jpg --pixel-diff --min-area 300 --blur 2.5 --added-thresh 25 --removed-thresh 25 --save-viz ./outputs/yolox_compare_viz.jpg --json --save ./outputs/yolox_compare.json
```

Flags:
- `--pixel-diff`: Enable pixel-based difference detection.
- `--min-area`: Minimum connected-component area to keep (filters noise).
- `--blur`: Gaussian blur radius before differencing (smooths noise).
- `--added-thresh` / `--removed-thresh`: Intensity thresholds for added/removed masks.
