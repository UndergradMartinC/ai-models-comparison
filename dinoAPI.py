import os
import json
import time
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    # Local project imports only; external runtime deps are imported lazily
    from grounding_dino import GroundingDINODetector, ConfusionMatrix, format_results
    from COCO_CLASSES import INDOOR_BUSINESS_CLASSES
    LOCAL_IMPORT_OK = True
except Exception as e:
    LOCAL_IMPORT_OK = False
    LOCAL_IMPORT_ERROR_MSG = str(e)


app = FastAPI(title="Grounding DINO API",
              version="1.0.0",
              description="API to run Grounding DINO evaluation with provided image and reference JSON")

# Open CORS for testing; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_upload_to_disk(upload: UploadFile, directory: str, target_filename: Optional[str] = None) -> str:
    os.makedirs(directory, exist_ok=True)
    filename = target_filename or upload.filename or "upload.bin"
    # Normalize filename (avoid path traversal)
    filename = os.path.basename(filename)
    path = os.path.join(directory, filename)
    with open(path, "wb") as out:
        # Stream to avoid loading into memory
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return path


def _parse_reference_json_bytes(data: bytes) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid reference JSON: {e}")

    # Normalize supported formats to list of objects
    if isinstance(payload, dict) and "objects" in payload:
        return payload["objects"]
    if isinstance(payload, list):
        return payload
    raise HTTPException(status_code=400, detail="Unsupported reference JSON format. Expected list or { 'objects': [...] }")


def _run_grounding_dino(image_path: str, reference_objects: List[Dict[str, Any]], use_gpu: bool, create_overlay: bool) -> Dict[str, Any]:
    if not LOCAL_IMPORT_OK:
        raise HTTPException(status_code=500, detail=f"Failed to import local modules: {LOCAL_IMPORT_ERROR_MSG}")

    # Create detector (loads model)
    device = "auto" if use_gpu else "cpu"
    detector = GroundingDINODetector(device=device)

    if detector.model is None:
        raise HTTPException(status_code=500, detail="Grounding DINO model not loaded. Check weights/config paths.")

    # Import external GroundingDINO utilities lazily to provide clearer errors
    try:
        from groundingdino.util.inference import load_image, predict
    except Exception as e:
        raise HTTPException(status_code=500, detail=(
            "Missing GroundingDINO runtime dependency. Install it first, e.g.: "
            "pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git' "
            "and ensure torch/torchvision are installed with CUDA if using GPU. "
            f"Original error: {e}"
        ))

    # Run prediction (replicates detect_objects_in_image and detector.detect_objects logic)
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape

    text_query = ". ".join(INDOOR_BUSINESS_CLASSES) + "."

    boxes, confidences, labels = predict(
        model=detector.model,
        image=image,
        caption=text_query,
        box_threshold=detector.box_threshold,
        text_threshold=detector.text_threshold,
        device=("cpu" if not use_gpu or str(detector.device) == "cpu" else "cuda"),
    )

    detections_for_cm: List[Dict[str, Any]] = []  # [{'class', 'bbox', 'confidence'}]
    detections_for_save: List[Dict[str, Any]] = []  # [{'object', 'bbox', 'confidence'}]

    for box, confidence, label in zip(boxes, confidences, labels):
        if confidence < detector.confidence_threshold:
            continue

        cx_norm, cy_norm, w_norm, h_norm = box
        cx = cx_norm * w
        cy = cy_norm * h
        box_w = w_norm * w
        box_h = h_norm * h

        x1 = int(cx - box_w / 2)
        y1 = int(cy - box_h / 2)
        x2 = int(cx + box_w / 2)
        y2 = int(cy + box_h / 2)

        # Clamp to bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        class_name = label.strip().lower()
        if class_name not in INDOOR_BUSINESS_CLASSES:
            continue

        detections_for_cm.append({
            "class": class_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(confidence),
        })
        detections_for_save.append({
            "object": class_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(confidence),
        })

    # Build confusion matrix from provided reference JSON
    matrix = ConfusionMatrix(reference_objects)
    for det in detections_for_cm:
        matrix.handle_object_data(det["class"], det["bbox"])

    class_metrics, mean_ap, mean_f1, mean_accuracy = matrix.get_matrix_metrics()
    metrics = format_results(class_metrics, mean_ap, mean_f1, mean_accuracy, matrix)
    # Remove verbose per-class metrics from API response
    metrics.pop("class_metrics", None)
    # Replace raw confusion matrix with a readable sparse representation
    raw_confusion = metrics.pop("confusion_matrix", None)
    if isinstance(raw_confusion, list):
        labels = INDOOR_BUSINESS_CLASSES
        non_zero_pairs: List[Dict[str, Any]] = []
        labels_used = set()
        for i, row in enumerate(raw_confusion):
            for j, val in enumerate(row):
                try:
                    count = int(val)
                except Exception:
                    # Fallback if values are floats
                    count = int(val) if val else 0
                if count > 0:
                    non_zero_pairs.append({
                        "true": labels[i],
                        "pred": labels[j],
                        "count": count,
                    })
                    labels_used.add(labels[i])
                    labels_used.add(labels[j])
        # Sort: diagonals first (TP), then by count desc
        non_zero_pairs.sort(key=lambda x: (x["true"] != x["pred"], -x["count"]))
        metrics["confusion_pairs"] = non_zero_pairs
        metrics["confusion_labels_used"] = sorted(labels_used)

    # Persist outputs (side-effects) but don't include in response
    try:
        detector.save_results(image_path, detections_for_save)
        if create_overlay:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            comparison_path = os.path.join("outputs", f"dino_{base_name}_comparison.jpg")
            detector.create_comparison_visualization(image_path, detections_for_save, matrix, comparison_path)
    except Exception:
        # Non-fatal for API response
        pass

    return {
        "comparison_results": metrics,
    }


@app.post("/run")
async def run_model(
    image: UploadFile = File(..., description="Image file to analyze"),
    reference_json: UploadFile = File(..., description="Reference JSON with ground truth annotations"),
    use_gpu: bool = Form(True),
    create_overlay: bool = Form(True),
):
    # Persist uploads
    uploads_dir = os.path.join("uploads")
    image_path = _read_upload_to_disk(image, uploads_dir)

    ref_bytes = await reference_json.read()
    reference_objects = _parse_reference_json_bytes(ref_bytes)

    _run_grounding_dino(image_path, reference_objects, use_gpu=use_gpu, create_overlay=create_overlay)

    # Execute
    started = time.time()
    try:
        result = _run_grounding_dino(image_path, reference_objects, use_gpu=use_gpu, create_overlay=create_overlay)
    finally:
        elapsed = time.time() - started

    response = {
        "status": "ok",
        "execution_time_seconds": elapsed,
        "comparison_results": result.get("comparison_results"),
    }
    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("dinoAPI:app", host=host, port=port, reload=False)


