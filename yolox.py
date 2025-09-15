import os
import random
import time
import hashlib
from typing import Any, Dict, List, Optional, TypedDict, cast


class Detection(TypedDict):
    class_name: str
    score: float
    bbox_xyxy: List[int]


def yolox() -> str:
    """Main YOLOX function used by the comparison harness in main.py.

    This simulates work to keep compatibility with the existing timing harness.
    """
    print("Running YOLOX model...")
    time.sleep(0.12)
    return "YOLOX completed"


def yolox_infer(
    image_path: Optional[str] = None,
    num_detections: int = 3,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a stubbed YOLOX inference for demo purposes.

    - Validates the optional image path (if provided)
    - Generates deterministic pseudo detections when seed is provided
    - Returns a structured dictionary with detections and timing

    This is a placeholder to be replaced by a real YOLOX inference pipeline later.
    """
    if image_path is not None and not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use deterministic per-image seed when not explicitly provided
    if seed is None and image_path is not None:
        seed = _seed_from_string(image_path)
    if seed is not None:
        random.seed(seed)

    start_time = time.time()
    # Simulate model load + inference latency
    time.sleep(0.12)

    # Try to use the real image size when available
    width = 1280
    height = 720
    if image_path is not None:
        try:
            from PIL import Image  # type: ignore
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception:
            # Fallback to default stub size if PIL unavailable or image unreadable
            pass

    candidate_classes = [
        "person",
        "car",
        "bicycle",
        "dog",
        "cat",
        "traffic light",
    ]

    detections: List[Detection] = []
    for _ in range(max(0, num_detections)):
        x1 = random.randint(0, max(0, width - 50))
        y1 = random.randint(0, max(0, height - 50))
        x2 = random.randint(x1 + 30, min(width, x1 + 300))
        y2 = random.randint(y1 + 30, min(height, y1 + 300))
        det: Detection = {
            "class_name": random.choice(candidate_classes),
            "score": round(random.uniform(0.3, 0.99), 3),
            "bbox_xyxy": [x1, y1, x2, y2],
        }
        detections.append(det)

    end_time = time.time()
    inference_ms = int((end_time - start_time) * 1000)

    return {
        "model_name": "YOLOX",
        "image_path": image_path,
        "image_size": {
            "width": width,
            "height": height,
        },
        "num_detections": len(detections),
        "detections": detections,
        "inference_ms": inference_ms,
    }


def yolox_compare(
    image_a_path: str,
    image_b_path: str,
    num_detections: int = 5,
    seed: Optional[int] = None,
    iou_threshold: float = 0.5,
    use_pixel_diff: bool = False,
    use_cv: bool = False,
    align: bool = False,
    min_area: int = 200,
    blur_radius: float = 2.0,
    added_thresh: int = 20,
    removed_thresh: int = 20,
    open_size: int = 3,
    close_size: int = 3,
    use_otsu: bool = False,
    thresh_offset: int = 0,
    merge_iou: float = 0.3,
) -> Dict[str, Any]:
    """Compare two images by running stubbed detections and diffing by IoU.

    Returns a dictionary with matches, additions in B, and removals from A.
    """
    if not os.path.exists(image_a_path):
        raise FileNotFoundError(f"Image A not found: {image_a_path}")
    if not os.path.exists(image_b_path):
        raise FileNotFoundError(f"Image B not found: {image_b_path}")

    if use_cv:
        start = time.time()
        diff = _cv_diff_boxes(
            image_a_path,
            image_b_path,
            align=align,
            min_area=min_area,
            blur_radius=blur_radius,
            use_otsu=use_otsu,
            thresh_offset=thresh_offset,
            open_size=open_size,
            close_size=close_size,
            merge_iou=merge_iou,
        )
        end = time.time()
        cv_added_in_b: List[Detection] = [
            {"class_name": "added", "score": 1.0, "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)]}
            for (x1, y1, x2, y2) in diff["added_boxes"]
        ]
        cv_removed_from_a: List[Detection] = [
            {"class_name": "removed", "score": 1.0, "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)]}
            for (x1, y1, x2, y2) in diff["removed_boxes"]
        ]
        summary = {
            "num_a": len(cv_removed_from_a),
            "num_b": len(cv_added_in_b),
            "num_matched": 0,
            "num_removed_from_a": len(cv_removed_from_a),
            "num_added_in_b": len(cv_added_in_b),
            "iou_threshold": float(iou_threshold),
        }
        return {
            "model_name": "YOLOX",
            "image_a": image_a_path,
            "image_b": image_b_path,
            "summary": summary,
            "matches": [],
            "removed_from_a": cv_removed_from_a,
            "added_in_b": cv_added_in_b,
            "a_inference_ms": int((end - start) * 1000),
            "b_inference_ms": int((end - start) * 1000),
        }

    if use_pixel_diff:
        start = time.time()
        diff = _pixel_diff_boxes(
            image_a_path,
            image_b_path,
            min_area=min_area,
            blur_radius=blur_radius,
            added_thresh=added_thresh,
            removed_thresh=removed_thresh,
            open_size=open_size,
            close_size=close_size,
            use_otsu=use_otsu,
            thresh_offset=thresh_offset,
            merge_iou=merge_iou,
        )
        end = time.time()
        # Package results in the same structure
        pd_added_in_b: List[Detection] = [
            {"class_name": "added", "score": 1.0, "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)]}
            for (x1, y1, x2, y2) in diff["added_boxes"]
        ]
        pd_removed_from_a: List[Detection] = [
            {"class_name": "removed", "score": 1.0, "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)]}
            for (x1, y1, x2, y2) in diff["removed_boxes"]
        ]
        summary = {
            "num_a": len(pd_removed_from_a),
            "num_b": len(pd_added_in_b),
            "num_matched": 0,
            "num_removed_from_a": len(pd_removed_from_a),
            "num_added_in_b": len(pd_added_in_b),
            "iou_threshold": float(iou_threshold),
        }
        return {
            "model_name": "YOLOX",
            "image_a": image_a_path,
            "image_b": image_b_path,
            "summary": summary,
            "matches": [],
            "removed_from_a": pd_removed_from_a,
            "added_in_b": pd_added_in_b,
            "a_inference_ms": int((end - start) * 1000),
            "b_inference_ms": int((end - start) * 1000),
        }

    # Derive deterministic seeds from inputs when not explicitly provided
    if seed is None:
        seed_a = _seed_from_string(image_a_path)
        seed_b = _seed_from_string(image_b_path)
    else:
        seed_a = seed
        seed_b = seed + 1

    result_a = yolox_infer(image_path=image_a_path, num_detections=num_detections, seed=seed_a)
    result_b = yolox_infer(image_path=image_b_path, num_detections=num_detections, seed=seed_b)

    dets_a = cast(List[Detection], result_a["detections"])
    dets_b = cast(List[Detection], result_b["detections"])

    matches: List[Dict[str, Any]] = []
    matched_b_indices: set[int] = set()

    # Greedy match A -> B by highest IoU above threshold
    for idx_a, det_a in enumerate(dets_a):
        best_iou: float = 0.0
        best_b_index: Optional[int] = None
        for idx_b, det_b in enumerate(dets_b):
            if idx_b in matched_b_indices:
                continue
            iou = _compute_iou(det_a["bbox_xyxy"], det_b["bbox_xyxy"])  # type: ignore[arg-type]
            if iou > best_iou:
                best_iou = iou
                best_b_index = idx_b
        if best_b_index is not None and best_iou >= iou_threshold:
            det_b = dets_b[best_b_index]
            matched_b_indices.add(best_b_index)
            matches.append(
                {
                    "iou": round(best_iou, 3),
                    "a": {
                        "class_name": det_a["class_name"],
                        "score": det_a["score"],
                        "bbox_xyxy": det_a["bbox_xyxy"],
                    },
                    "b": {
                        "class_name": det_b["class_name"],
                        "score": det_b["score"],
                        "bbox_xyxy": det_b["bbox_xyxy"],
                    },
                    "class_changed": det_a["class_name"] != det_b["class_name"],
                }
            )

    removed_from_a: List[Detection] = []
    for idx_a, det_a in enumerate(dets_a):
        # If det_a not in matches' a entries, it's removed
        if not any(_same_bbox(det_a["bbox_xyxy"], m["a"]["bbox_xyxy"]) for m in matches):
            removed_from_a.append(det_a)

    added_in_b: List[Detection] = []
    for idx_b, det_b in enumerate(dets_b):
        if idx_b not in matched_b_indices:
            added_in_b.append(det_b)

    summary = {
        "num_a": len(dets_a),
        "num_b": len(dets_b),
        "num_matched": len(matches),
        "num_removed_from_a": len(removed_from_a),
        "num_added_in_b": len(added_in_b),
        "iou_threshold": iou_threshold,
    }

    return {
        "model_name": "YOLOX",
        "image_a": image_a_path,
        "image_b": image_b_path,
        "summary": summary,
        "matches": matches,
        "removed_from_a": removed_from_a,
        "added_in_b": added_in_b,
        "a_inference_ms": result_a["inference_ms"],  # type: ignore[index]
        "b_inference_ms": result_b["inference_ms"],  # type: ignore[index]
    }


def _compute_iou(box1: List[int], box2: List[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return 0.0 if union == 0 else inter_area / union


def _same_bbox(b1: List[int], b2: List[int]) -> bool:
    return tuple(b1) == tuple(b2)


def _seed_from_string(s: str) -> int:
    # Stable deterministic seed from string
    digest = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pixel_diff_boxes(
    image_a_path: str,
    image_b_path: str,
    min_area: int = 200,
    blur_radius: float = 2.0,
    added_thresh: int = 20,
    removed_thresh: int = 20,
    open_size: int = 3,
    close_size: int = 3,
    use_otsu: bool = False,
    thresh_offset: int = 0,
    merge_iou: float = 0.3,
) -> Dict[str, List[tuple[int, int, int, int]]]:
    """Compute simple pixel-diff boxes for added/removed regions.

    Heuristic approach using Pillow only (no heavy CV deps):
    - Convert to grayscale
    - Gaussian blur to reduce noise
    - Compute absolute difference A->B and B->A
    - Threshold and extract connected components via a simple BFS over binary mask
    """
    from PIL import Image, ImageFilter, ImageChops

    with Image.open(image_a_path) as ia, Image.open(image_b_path) as ib:
        ia = ia.convert("L")
        ib = ib.convert("L")
        if ia.size != ib.size:
            ib = ib.resize(ia.size)

        ia_blur = ia.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        ib_blur = ib.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        diff_img = ImageChops.difference(ib_blur, ia_blur)  # B - A (added)
        diff_img_rev = ImageChops.difference(ia_blur, ib_blur)  # A - B (removed)

        def to_mask(img: Image.Image, base_thresh: int) -> List[List[int]]:
            # Optionally compute Otsu threshold
            t = base_thresh
            if use_otsu:
                t = max(0, min(255, _otsu_threshold_from_image(img) + thresh_offset))
            # Threshold to binary
            bw = img.point(lambda p: 255 if p >= t else 0, mode='L')
            # Morphology: opening then closing if requested sizes are valid odd ints >= 3
            if open_size and open_size >= 3 and open_size % 2 == 1:
                bw = bw.filter(ImageFilter.MinFilter(open_size))
                bw = bw.filter(ImageFilter.MaxFilter(open_size))
            if close_size and close_size >= 3 and close_size % 2 == 1:
                bw = bw.filter(ImageFilter.MaxFilter(close_size))
                bw = bw.filter(ImageFilter.MinFilter(close_size))
            w, h = bw.size
            px = bw.load()
            arr = [[0] * h for _ in range(w)]
            for x in range(w):
                for y in range(h):
                    arr[x][y] = 255 if px[x, y] > 0 else 0
            return arr

        diff_ab = to_mask(diff_img, added_thresh)
        diff_ba = to_mask(diff_img_rev, removed_thresh)

        added_boxes = _extract_boxes_from_mask(diff_ab, threshold=1, min_area=min_area)
        removed_boxes = _extract_boxes_from_mask(diff_ba, threshold=1, min_area=min_area)

        # Merge overlapping/adjacent boxes
        if merge_iou > 0:
            added_boxes = _merge_overlapping_boxes(added_boxes, iou_threshold=merge_iou)
            removed_boxes = _merge_overlapping_boxes(removed_boxes, iou_threshold=merge_iou)

        return {
            "added_boxes": added_boxes,
            "removed_boxes": removed_boxes,
        }


def _extract_boxes_from_mask(
    diff: List[List[int]], threshold: int, min_area: int
) -> List[tuple[int, int, int, int]]:
    width = len(diff)
    height = len(diff[0]) if width > 0 else 0
    visited = [[False] * height for _ in range(width)]

    boxes: List[tuple[int, int, int, int]] = []
    for x in range(width):
        for y in range(height):
            if visited[x][y] or diff[x][y] < threshold:
                continue
            # BFS to get connected component
            stack = [(x, y)]
            visited[x][y] = True
            min_x = x
            min_y = y
            max_x = x
            max_y = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                if cx < min_x:
                    min_x = cx
                if cy < min_y:
                    min_y = cy
                if cx > max_x:
                    max_x = cx
                if cy > max_y:
                    max_y = cy
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if nx == cx and ny == cy:
                            continue
                        if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny] and diff[nx][ny] >= threshold:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
            if area >= min_area:
                # Expand by 1 pixel margin
                boxes.append(
                    (
                        max(0, min_x - 1),
                        max(0, min_y - 1),
                        min(width - 1, max_x + 1),
                        min(height - 1, max_y + 1),
                    )
                )

    return boxes


def _cv_diff_boxes(
    image_a_path: str,
    image_b_path: str,
    align: bool = False,
    min_area: int = 200,
    blur_radius: float = 2.0,
    use_otsu: bool = False,
    thresh_offset: int = 0,
    open_size: int = 3,
    close_size: int = 3,
    merge_iou: float = 0.3,
) -> Dict[str, List[tuple[int, int, int, int]]]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV not available. Install opencv-python-headless.") from e

    img_a = cv2.imdecode(np.fromfile(image_a_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np.fromfile(image_b_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_a is None or img_b is None:
        # Fallback to PIL path if cv load fails
        raise RuntimeError("Failed to load images with OpenCV.")

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))

    if align:
        # ECC alignment (translation + rotation + scale via homography)
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            cc, warp_matrix = cv2.findTransformECC(gray_a, gray_b, warp_matrix, warp_mode, criteria)
            gray_b = cv2.warpAffine(gray_b, warp_matrix, (gray_a.shape[1], gray_a.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            pass

    # Directional diffs
    diff_ab = cv2.subtract(gray_b, gray_a)
    diff_ba = cv2.subtract(gray_a, gray_b)

    def process_diff(diff_img):
        ksize = max(1, int(round(blur_radius)))
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(diff_img, (ksize, ksize), 0)
        if use_otsu:
            thr, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if thresh_offset != 0:
                thr = max(0, min(255, int(thr) + int(thresh_offset)))
                _, mask = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(blurred, max(1, int(thresh_offset)), 255, cv2.THRESH_BINARY)

        # Morphology
        if open_size and open_size >= 3 and open_size % 2 == 1:
            kernel_o = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_o)
        if close_size and close_size >= 3 and close_size % 2 == 1:
            kernel_c = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_c)
        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area:
                boxes.append((x, y, x + w, y + h))
        return boxes

    added_boxes = process_diff(diff_ab)
    removed_boxes = process_diff(diff_ba)

    if merge_iou > 0:
        added_boxes = _merge_overlapping_boxes(added_boxes, iou_threshold=merge_iou)
        removed_boxes = _merge_overlapping_boxes(removed_boxes, iou_threshold=merge_iou)

    return {"added_boxes": added_boxes, "removed_boxes": removed_boxes}


def _merge_overlapping_boxes(
    boxes: List[tuple[int, int, int, int]], iou_threshold: float = 0.3
) -> List[tuple[int, int, int, int]]:
    if not boxes:
        return boxes
    merged = True
    current = boxes[:]
    while merged:
        merged = False
        new_boxes: List[tuple[int, int, int, int]] = []
        used = [False] * len(current)
        for i in range(len(current)):
            if used[i]:
                continue
            x1a, y1a, x2a, y2a = current[i]
            used[i] = True
            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                iou = _compute_iou([x1a, y1a, x2a, y2a], [current[j][0], current[j][1], current[j][2], current[j][3]])
                if iou >= iou_threshold:
                    used[j] = True
                    x1a = min(x1a, current[j][0])
                    y1a = min(y1a, current[j][1])
                    x2a = max(x2a, current[j][2])
                    y2a = max(y2a, current[j][3])
                    merged = True
            new_boxes.append((x1a, y1a, x2a, y2a))
        current = new_boxes
    return current


def _otsu_threshold_from_image(img) -> int:
    # Compute Otsu threshold from a grayscale PIL image
    hist = img.histogram()[:256]
    total = sum(hist)
    sum_total = sum(i * h for i, h in enumerate(hist))
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold