import argparse
import json
from pathlib import Path
from typing import Any, Dict

from yolox import yolox_compare, yolox_infer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOX standalone demo (stub)")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to an input image (validated if provided)",
    )
    parser.add_argument(
        "--image-a",
        type=str,
        default=None,
        help="Path to image A for comparison mode",
    )
    parser.add_argument(
        "--image-b",
        type=str,
        default=None,
        help="Path to image B for comparison mode",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        help="Number of fake detections to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic outputs",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching in comparison mode",
    )
    parser.add_argument(
        "--pixel-diff",
        action="store_true",
        help="Use pixel-diff comparison instead of random detections",
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Use OpenCV-based alignment + diff pipeline",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Align images before diff (when using --use-cv)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Minimum blob area for pixel-diff boxes",
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=2.0,
        help="Gaussian blur radius to reduce noise for pixel-diff",
    )
    parser.add_argument(
        "--added-thresh",
        type=int,
        default=20,
        help="Threshold for additions mask in pixel-diff",
    )
    parser.add_argument(
        "--removed-thresh",
        type=int,
        default=20,
        help="Threshold for removals mask in pixel-diff",
    )
    parser.add_argument(
        "--use-otsu",
        action="store_true",
        help="Use Otsu thresholding for pixel-diff (overrides added/removed thresholds, can be nudged with --thresh-offset)",
    )
    parser.add_argument(
        "--thresh-offset",
        type=int,
        default=0,
        help="Offset added to Otsu threshold (can be negative)",
    )
    parser.add_argument(
        "--open-size",
        type=int,
        default=3,
        help="Morphological opening kernel size (odd >= 3)",
    )
    parser.add_argument(
        "--close-size",
        type=int,
        default=3,
        help="Morphological closing kernel size (odd >= 3)",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.3,
        help="IoU threshold to merge overlapping pixel-diff boxes",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results in JSON format",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path to save JSON results",
    )
    parser.add_argument(
        "--save-viz",
        type=str,
        default=None,
        help="Optional output path to save annotated comparison image (comparison mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Comparison mode when both images are given
    if args.image_a and args.image_b:
        image_a = str(Path(args.image_a).expanduser())
        image_b = str(Path(args.image_b).expanduser())
        result: Dict[str, Any] = yolox_compare(
            image_a_path=image_a,
            image_b_path=image_b,
            num_detections=args.num,
            seed=args.seed,
            iou_threshold=args.iou,
            use_pixel_diff=args.pixel_diff,
            use_cv=args.use_cv,
            align=args.align,
            min_area=args.min_area,
            blur_radius=args.blur,
            added_thresh=args.added_thresh,
            removed_thresh=args.removed_thresh,
            open_size=args.open_size,
            close_size=args.close_size,
            use_otsu=args.use_otsu,
            thresh_offset=args.thresh_offset,
            merge_iou=args.merge_iou,
        )
        if args.json:
            output = json.dumps(result, indent=2)
        else:
            s = result["summary"]
            lines = [
                f"Model: {result['model_name']}",
                f"Image A: {result['image_a']}",
                f"Image B: {result['image_b']}",
                f"IoU threshold: {s['iou_threshold']}",
                f"A detections: {s['num_a']}  B detections: {s['num_b']}",
                f"Matches: {s['num_matched']}  Removed(A): {s['num_removed_from_a']}  Added(B): {s['num_added_in_b']}",
                f"A inference: {result['a_inference_ms']} ms  B inference: {result['b_inference_ms']} ms",
                "",
                "Matches:",
            ]
            for idx, m in enumerate(result["matches" ]):
                a = m["a"]
                b = m["b"]
                lines.append(
                    f"  #{idx+1}: IoU={m['iou']:.3f}  A({a['class_name']}, {a['score']:.3f}) -> B({b['class_name']}, {b['score']:.3f})"
                )
            if result["removed_from_a"]:
                lines.append("\nRemoved from A:")
                for det in result["removed_from_a"]:
                    lines.append(f"  - {det['class_name']}  score={det['score']:.3f}  bbox={det['bbox_xyxy']}")
            if result["added_in_b"]:
                lines.append("\nAdded in B:")
                for det in result["added_in_b"]:
                    lines.append(f"  + {det['class_name']}  score={det['score']:.3f}  bbox={det['bbox_xyxy']}")
            output = "\n".join(lines)
        print(output)
        # Save JSON if requested
        if args.save is not None:
            save_path = Path(args.save).expanduser()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(result, indent=2))
            print(f"Saved results to: {str(save_path)}")

        # Save visualization if requested
        if args.save_viz is not None:
            try:
                from PIL import Image, ImageDraw  # type: ignore
                image_b_path = Path(result["image_b"]).expanduser()
                with Image.open(image_b_path) as im:
                    draw = ImageDraw.Draw(im)
                    # Added in B -> green boxes
                    for det in result["added_in_b"]:
                        x1, y1, x2, y2 = det["bbox_xyxy"]
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                    # Removed from A -> red boxes (draw on B image for a single output)
                    for det in result["removed_from_a"]:
                        x1, y1, x2, y2 = det["bbox_xyxy"]
                        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                    # Optionally thin outlines for matches where class changed: yellow
                    for m in result["matches"]:
                        if m.get("class_changed"):
                            bx1, by1, bx2, by2 = m["b"]["bbox_xyxy"]
                            draw.rectangle([bx1, by1, bx2, by2], outline=(255, 215, 0), width=2)
                    viz_path = Path(args.save_viz).expanduser()
                    viz_path.parent.mkdir(parents=True, exist_ok=True)
                    im.save(viz_path)
                    print(f"Saved visualization to: {str(viz_path)}")
            except Exception as e:
                print(f"Failed to save visualization: {e}")
        return

    # Single-image inference mode
    image_path = args.image
    if image_path is not None:
        image_path = str(Path(image_path).expanduser())

    result2: Dict[str, Any] = yolox_infer(
        image_path=image_path,
        num_detections=args.num,
        seed=args.seed,
    )

    if args.json:
        output2 = json.dumps(result2, indent=2)
    else:
        lines2 = [
            f"Model: {result2['model_name']}",
            f"Image: {result2['image_path']}",
            f"Image size: {result2['image_size']['width']}x{result2['image_size']['height']}",
            f"Detections: {result2['num_detections']}",
            f"Inference: {result2['inference_ms']} ms",
        ]
        for idx, det in enumerate(result2["detections"]):
            cls = det["class_name"]
            score = det["score"]
            x1, y1, x2, y2 = det["bbox_xyxy"]
            lines2.append(
                f"  #{idx+1}: {cls:>12s}  score={score:.3f}  bbox=[{x1}, {y1}, {x2}, {y2}]"
            )
        output2 = "\n".join(lines2)

    print(output2)

    if args.save is not None:
        save_path = Path(args.save).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(result2, indent=2))
        print(f"Saved results to: {str(save_path)}")


if __name__ == "__main__":
    main()


