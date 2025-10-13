"""
COCO Bounding Box Annotator & JSON Exporter

What this script does
---------------------
• Lets you draw bounding boxes on your own images (click + drag) and assign a COCO class to each box.
• Saves annotations per-image as VALID JSON with format:
  [
    { "class": "person", "bbox": [x0, y0, x1, y1] },
    { "class": "chair", "bbox": [x0, y0, x1, y1] }
  ]

How to use
----------
1) Install requirements (Python 3.9+ recommended):
   pip install opencv-python pillow

2) Put your images in a folder, e.g., ./images

3) Run the script:
   python annotator.py --images ./images --out ./labels

4) Controls inside the annotation window:
   • Left mouse: click+drag to draw a box
   • After releasing the mouse: you'll be prompted in the terminal to enter a COCO class id or name
   • u : undo last box on this image
   • n : next image (saves current image annotations automatically)
   • p : previous image (current annotations are saved first)
   • s : save annotations for current image
   • q : save and quit

Notes
-----
• Coordinates are saved as integers in image pixel space.
• Boxes are (x0, y0) top-left and (x1, y1) bottom-right, automatically ordered even if you drag in reverse.
• COCO class ids (0–79) and names are provided below.
• The script keeps a sidecar JSON per image, e.g., 0001.jpg -> 0001.json

"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2

# ============================
# Configuration
# ============================
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
ACTIVE_BOX_COLOR = (0, 200, 255)

# Official COCO-2017 80-class list in order (id == index)
COCO_CLASSES = [
    "person",          # People in the space
    "backpack",        # Luggage/personal items
    "handbag",         # Personal bags
    "tie",             # Professional attire
    "suitcase",        # Travel luggage (Airbnb guests)
    "bottle",          # Water bottles, beverages
    "wine glass",      # Dining/entertainment
    "cup",             # Coffee cups, mugs
    "fork",            # Dining utensils
    "knife",           # Kitchen utensils
    "spoon",           # Dining utensils
    "bowl",            # Dishes
    "banana",          # Food items
    "apple",           # Food items
    "sandwich",        # Food items
    "chair",           # Office/dining furniture
    "couch",           # Lounge furniture
    "potted plant",    # Decor/ambiance
    "bed",             # Bedroom furniture
    "dining table",    # Dining/meeting furniture
    "toilet",          # Bathroom fixtures
    "tv",              # Entertainment/presentation
    "laptop",          # Work equipment
    "mouse",           # Computer peripherals
    "remote",          # TV/AC controls
    "keyboard",        # Computer equipment
    "cell phone",      # Personal devices
    "microwave",       # Kitchen appliances
    "oven",            # Kitchen appliances
    "toaster",         # Kitchen appliances
    "sink",            # Kitchen/bathroom fixtures
    "refrigerator",    # Kitchen appliances
    "book",            # Reading materials/decor
    "clock",           # Time displays
    "vase",            # Decorative items
    "scissors",        # Office supplies
    "teddy bear",      # Comfort items (hotels)
    "hair drier",      # Bathroom amenities
    "toothbrush"       # Personal hygiene items
]

# ============================
# Data structures
# ============================
class Annotation:
    def __init__(self, cls_name: str, tl: Tuple[int, int], br: Tuple[int, int]):
        self.cls_name = cls_name
        self.tl = tl
        self.br = br

    def as_json(self) -> Dict:
        return {
            "class": self.cls_name,
            "bbox": [int(self.tl[0]), int(self.tl[1]), int(self.br[0]), int(self.br[1])]
        }

# ============================
# Utility functions
# ============================

def clamp_box(x0, y0, x1, y1, w, h):
    x0, x1 = sorted([int(round(x0)), int(round(x1))])
    y0, y1 = sorted([int(round(y0)), int(round(y1))])
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    return x0, y0, x1, y1


def class_from_user(raw: str) -> str:
    raw = raw.strip()
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(COCO_CLASSES):
            return COCO_CLASSES[idx]
        else:
            print(f"[!] Class id {idx} out of range 0..{len(COCO_CLASSES)-1}. Try again.")
            return ""
    lower = raw.lower()
    for i, name in enumerate(COCO_CLASSES):
        if name.lower() == lower:
            return name
    print(f"[!] Unknown class name '{raw}'. Try again.")
    return ""

# ============================
# Annotator
# ============================
class ImageAnnotator:
    def __init__(self, image_paths: List[Path], out_dir: Path):
        self.image_paths = image_paths
        self.out_dir = out_dir
        self.idx = 0
        self.annos: Dict[str, List[Annotation]] = {str(p): [] for p in image_paths}
        self.dragging = False
        self.start_pt = (0, 0)
        self.current_img = None
        self.current_vis = None
        self.window = 'Annotator'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse_cb)

    def _load(self):
        path = self.image_paths[self.idx]
        self.current_img = cv2.imread(str(path))
        if self.current_img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        self._refresh()

    def _refresh(self):
        img = self.current_img.copy()
        for a in self.annos[str(self.image_paths[self.idx])]:
            cv2.rectangle(img, a.tl, a.br, BOX_COLOR, BOX_THICKNESS)
            cv2.putText(img, a.cls_name, (a.tl[0], max(0, a.tl[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
        self.current_vis = img
        cv2.imshow(self.window, img)
        cv2.setWindowTitle(self.window, f"{self.image_paths[self.idx].name}  [n: next, p: prev, u: undo, s: save, q: quit]")

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            img = self.current_img.copy()
            x0, y0, x1, y1 = clamp_box(self.start_pt[0], self.start_pt[1], x, y, img.shape[1], img.shape[0])
            cv2.rectangle(img, (x0, y0), (x1, y1), ACTIVE_BOX_COLOR, BOX_THICKNESS)
            cv2.imshow(self.window, img)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x0, y0, x1, y1 = clamp_box(self.start_pt[0], self.start_pt[1], x, y, self.current_img.shape[1], self.current_img.shape[0])
            if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
                self._refresh()
                return
            while True:
                user = input("Enter COCO class id (0-79) or name for this box: ")
                cls = class_from_user(user)
                if cls:
                    break
            self.annos[str(self.image_paths[self.idx])].append(Annotation(cls, (x0, y0), (x1, y1)))
            self._refresh()

    def _save_current(self):
        path = self.image_paths[self.idx]
        anns = self.annos[str(path)]
        data = [a.as_json() for a in anns]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.out_dir / f"{path.stem}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {json_path}")

    def run(self):
        self._load()
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('u'):
                if self.annos[str(self.image_paths[self.idx])]:
                    self.annos[str(self.image_paths[self.idx])].pop()
                    print("Undid last box.")
                    self._refresh()
            elif key == ord('s'):
                self._save_current()
            elif key == ord('n'):
                self._save_current()
                if self.idx < len(self.image_paths) - 1:
                    self.idx += 1
                    self._load()
                else:
                    print("[End] No more images. Press q to quit or p to go back.")
            elif key == ord('p'):
                self._save_current()
                if self.idx > 0:
                    self.idx -= 1
                    self._load()
                else:
                    print("[Start] This is the first image.")
            elif key == ord('q'):
                self._save_current()
                print("Quitting.")
                break
        cv2.destroyAllWindows()

# ============================
# CLI
# ============================

def collect_images(folder: Path) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    paths = [p for p in sorted(folder.rglob('*')) if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found under: {folder}")
    return paths


def main():
    parser = argparse.ArgumentParser(description='COCO Bounding Box Annotator')
    parser.add_argument('--images', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--out', type=str, required=True, help='Output folder for JSON labels')
    args = parser.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.out)
    imgs = collect_images(img_dir)

    print("Loaded", len(imgs), "images.")
    print("COCO classes (id:name):")
    for i, n in enumerate(COCO_CLASSES):
        print(f"{i:2d}: {n}")

    annotator = ImageAnnotator(imgs, out_dir)
    annotator.run()


if __name__ == '__main__':
    main()
