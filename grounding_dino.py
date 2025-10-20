"""
Grounding DINO Implementation
Open vocabulary object detection using text prompts
"""

import cv2
import numpy as np
import json
import os
import warnings
from typing import List, Dict
from collections import Counter
from model_tests import ConfusionMatrix
from COCO_CLASSES import INDOOR_BUSINESS_CLASSES

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Check for Grounding DINO availability
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    GROUNDINGDINO_AVAILABLE = True
    print("[OK] Grounding DINO available!")
except ImportError:
    GROUNDINGDINO_AVAILABLE = False
    print("[WARNING]  Grounding DINO not available")


class GroundingDINODetector:
    """Grounding DINO for open vocabulary detection"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.box_threshold = 0.5
        self.text_threshold = 0.5
        self.confidence_threshold = 0.5
        
        self._load_model()
    
    def _get_device(self, device: str):
        """Get appropriate device"""
        if device == "auto":
            import torch
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load Grounding DINO model"""
        if not GROUNDINGDINO_AVAILABLE:
            print("[WARNING]  Grounding DINO not available")
            return False
        
        try:
            weights_path = "weights/groundingdino_swint_ogc.pth"
            config_path = "weights/GroundingDINO_SwinT_OGC.py"
            
            if not all(os.path.exists(p) for p in [weights_path, config_path]):
                print("[ERROR] Grounding DINO model files not found")
                return False
            
            self.model = load_model(config_path, weights_path, device="cpu")
            print("[OK] Grounding DINO loaded successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load Grounding DINO: {e}")
            return False
    
    def get_comprehensive_prompts(self) -> List[str]:
        """Get comprehensive list of object prompts"""
        return [
            # Furniture
            "chair", "table", "desk", "shelf", "cabinet", "drawer", "bookshelf",
            "couch", "sofa", "stool", "bench", "furniture",
            
            # Technology
            "computer", "laptop", "monitor", "screen", "keyboard", "mouse", 
            "printer", "phone", "tablet", "cable",
            
            # Office supplies
            "book", "paper", "document", "folder", "pen", "pencil", "notebook",
            
            # Storage & containers
            "box", "bag", "backpack", "basket", "bin", "trash can", "container", "storage",
            
            # Lighting & electrical
            "lamp", "light", "bulb",
            
            # Decorative & misc
            "plant", "flower", "picture", "painting", "frame", "mirror", "clock",
            
            # Food & drinks
            "bottle", "cup", "mug", "glass", "water",
            
            # Room elements
            "window", "door", "wall", "floor"

            #mess
            "spill", "stain", "mess"
        ]
    
    def detect_objects(self, image_path: str, custom_prompts: List[str] = None) -> List[Dict]:
        """Detect objects using Grounding DINO"""
        if self.model is None:
            print("[ERROR] Grounding DINO model not loaded")
            return []
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return []
        
        try:
            print(f" Running Grounding DINO on {image_path}")
            
            # Load image
            image_source, image = load_image(image_path)
            h, w, _ = image_source.shape
            
            # Use custom prompts or INDOOR_BUSINESS_CLASSES
            prompts = custom_prompts if custom_prompts else INDOOR_BUSINESS_CLASSES
            
            # Create text query
            text_query = ". ".join(prompts) + "."
            
            # Run detection
            boxes, confidences, labels = predict(
                model=self.model,
                image=image,
                caption=text_query,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device="cpu"
            )
            
            # Convert coordinates: [cx, cy, w, h] -> [x1, y1, x2, y2]
            detections = []
            for box, confidence, label in zip(boxes, confidences, labels):
                if confidence >= self.confidence_threshold:
                    cx_norm, cy_norm, w_norm, h_norm = box
                    
                    cx = cx_norm * w
                    cy = cy_norm * h
                    box_w = w_norm * w
                    box_h = h_norm * h
                    
                    x1 = int(cx - box_w / 2)
                    y1 = int(cy - box_h / 2)
                    x2 = int(cx + box_w / 2)
                    y2 = int(cy + box_h / 2)
                    
                    # Clamp to image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # Validate and filter to only INDOOR_BUSINESS_CLASSES
                    if x2 > x1 and y2 > y1:
                        class_name = label.strip().lower()
                        # Only include objects that are in INDOOR_BUSINESS_CLASSES
                        if class_name in INDOOR_BUSINESS_CLASSES:
                            detection = {
                                "object": class_name,
                                "confidence": float(confidence),
                                "bbox": [x1, y1, x2, y2],
                                "detection_type": "grounding_dino"
                            }
                            detections.append(detection)
            
            print(f"[OK] Grounding DINO detected {len(detections)} objects")

            # Create matrix from JSON ground truth
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ground_truth_file = f"test_photos/labels/{base_name}_test.json"

            # Fallback to regular json file if _test.json doesn't exist
            if not os.path.exists(ground_truth_file):
                ground_truth_file = f"test_photos/labels/{base_name}.json"

            # Final fallback to sample_test.json in current directory
            if not os.path.exists(ground_truth_file):
                ground_truth_file = "sample_test.json"

            if os.path.exists(ground_truth_file):
                with open(ground_truth_file, 'r') as f:
                    ground_truth = json.load(f)
                
                # Handle different JSON formats
                # Format 1: [{"class": "...", "bbox": [...]}, ...]
                # Format 2: {"image": "...", "objects": [{"class": "...", "bbox": [...]}]}
                if isinstance(ground_truth, dict) and 'objects' in ground_truth:
                    ground_truth = ground_truth['objects']
                
                matrix = ConfusionMatrix(ground_truth)
                print(f" Created confusion matrix from {ground_truth_file}")
            else:
                print(f"[WARNING] Ground truth file not found: {ground_truth_file}")
                matrix = None

            # Handle each detection through confusion matrix
            if matrix is not None:
                for detection in detections:
                    detected_class = detection["object"]
                    bbox = detection["bbox"]
                    matrix.handle_object_data(detected_class, bbox)

            # Get matrix metrics
            if matrix is not None:
                class_metrics, mean_ap, mean_f1, mean_accuracy = matrix.get_matrix_metrics()
                print(f" Confusion Matrix Results: mAP={mean_ap:.3f}, mF1={mean_f1:.3f}")

                # Print per-class metrics for detected classes
                detected_classes = set(d["object"] for d in detections)
                for metric in class_metrics:
                    if metric.class_name in detected_classes:
                        print(f"  {metric.class_name}: P={metric.precision:.3f}, R={metric.sensitivity:.3f}, F1={metric.f1_score:.3f}")

            # Save comparison visualization image (shows both detections and ground truth)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            vis_path = f"outputs/dino_{base_name}_comparison.jpg"
            self.create_comparison_visualization(image_path, detections, matrix, vis_path)

            return detections
            
        except Exception as e:
            print(f"[ERROR] Grounding DINO detection failed: {e}")
            return []
    
    def create_visualization(self, image_path: str, detections: List[Dict], save_path: str = None):
        """Create visualization of detections"""
        if not os.path.exists(image_path):
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Color scheme for different object types
        colors = {
            "chair": (0, 255, 0),      # Green
            "table": (255, 0, 0),      # Red
            "desk": (255, 0, 0),       # Red
            "computer": (0, 0, 255),   # Blue
            "monitor": (0, 0, 255),    # Blue
            "laptop": (0, 0, 255),     # Blue
            "lamp": (255, 255, 0),     # Yellow
            "plant": (0, 255, 255),    # Cyan
            "book": (255, 0, 255),     # Magenta
            "bag": (128, 0, 128),      # Purple
            "bottle": (255, 165, 0),   # Orange
        }
        
        # Draw detections
        for det in detections:
            bbox = det["bbox"]
            obj = det["object"]
            conf = det["confidence"]
            
            # Choose color
            color = colors.get(obj.split()[0], (128, 128, 128))  # Default gray
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box (thick for DINO)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f" {obj}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Position label
            text_x = x1
            text_y = y1 - 5
            if text_y < 15:
                text_y = y1 + 20
            
            # Text background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image_rgb, (text_x, text_y - text_height - 3),
                         (text_x + text_width, text_y + 3), color, -1)
            
            # White text
            cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Save if path provided
        if save_path:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)
            print(f" Visualization saved: {save_path}")
        
        return image_rgb
    
    def create_comparison_visualization(self, image_path: str, detections: List[Dict], matrix, save_path: str = None):
        """Create visualization showing both model detections and ground truth annotations"""
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return None

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Colors for different types of boxes
        colors = {
            "detection": (0, 255, 0),      # Green for model detections
            "ground_truth": (255, 0, 0),   # Red for ground truth
            "missing": (0, 0, 255),       # Blue for missing ground truth
            "false_positive": (255, 255, 0)  # Yellow for false positives
        }

        # Draw model detections (green boxes)
        for det in detections:
            bbox = det["bbox"]
            obj = det.get("object", det.get("class", "unknown"))  # Handle both 'object' and 'class' keys
            conf = det["confidence"]

            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), colors["detection"], 2)

            # Draw label
            label = f" {obj}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Position label
            text_x = x1
            text_y = y1 - 5
            if text_y < 15:
                text_y = y1 + 20

            # Text background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image_rgb, (text_x, text_y - text_height - 3),
                         (text_x + text_width, text_y + 3), colors["detection"], -1)

            # White text
            cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Draw ground truth annotations (red boxes)
        ground_truth_objects = matrix.reference_object_array
        for gt_obj in ground_truth_objects:
            bbox = gt_obj.bbox
            class_name = gt_obj.class_name

            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), colors["ground_truth"], 2)

            # Draw label
            label = f" GT: {class_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Position label
            text_x = x1
            text_y = y2 + 15  # Position below box to avoid overlap with detection labels
            if text_y > image_rgb.shape[0] - 5:
                text_y = y2 - 5

            # Text background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image_rgb, (text_x, text_y - text_height - 3),
                         (text_x + text_width, text_y + 3), colors["ground_truth"], -1)

            # White text
            cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Draw missing ground truth objects (blue boxes)
        missing_objects = matrix.missing_objects
        for missing_obj in missing_objects:
            bbox = missing_obj.bbox
            class_name = missing_obj.class_name

            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), colors["missing"], 2)

            # Draw label
            label = f"[ERROR] MISSING: {class_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Position label
            text_x = x1
            text_y = y1 - 5
            if text_y < 15:
                text_y = y1 + 20

            # Text background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image_rgb, (text_x, text_y - text_height - 3),
                         (text_x + text_width, text_y + 3), colors["missing"], -1)

            # White text
            cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Add legend
        legend_y = 20
        legend_items = [
            (" Model Detection", colors["detection"]),
            (" Ground Truth", colors["ground_truth"]),
            ("[ERROR] Missing GT", colors["missing"])
        ]

        for text, color in legend_items:
            cv2.putText(image_rgb, text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            legend_y += 25

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)
            print(f" Comparison visualization saved: {save_path}")

        return image_rgb
    
    def analyze_detections(self, detections: List[Dict]) -> Dict:
        """Analyze Grounding DINO detections"""
        objects = [det["object"] for det in detections]
        
        # Define categories for Grounding DINO prompts
        categories = {
            "furniture": ["chair", "table", "desk", "shelf", "cabinet", "furniture", "bookshelf", "couch", "sofa", "stool", "bench"],
            "technology": ["computer", "laptop", "monitor", "screen", "keyboard", "mouse", "printer", "phone", "tablet", "cable"],
            "supplies": ["book", "paper", "document", "folder", "pen", "pencil", "notebook"],
            "storage": ["box", "bag", "backpack", "basket", "bin", "container", "storage"],
            "lighting": ["lamp", "light", "bulb"],
            "decorative": ["plant", "flower", "picture", "painting", "frame", "mirror", "clock"],
            "personal": ["bottle", "cup", "mug", "glass", "water"],
            "room_elements": ["window", "door", "wall", "floor"]
        }
        
        # Count objects by category
        category_counts = {}
        for category, keywords in categories.items():
            count = sum(1 for obj in objects if any(keyword in obj for keyword in keywords))
            if count > 0:
                category_counts[category] = count
        
        # Get object counts
        object_counts = Counter(objects)
        
        return {
            "total_objects": len(detections),
            "categories": category_counts,
            "object_counts": dict(object_counts.most_common()),
            "unique_objects": len(set(objects)),
            "confidence_stats": {
                "min": min([det["confidence"] for det in detections]) if detections else 0,
                "max": max([det["confidence"] for det in detections]) if detections else 0,
                "avg": sum([det["confidence"] for det in detections]) / len(detections) if detections else 0
            }
        }
    
    def save_results(self, image_path: str, detections: List[Dict], output_dir: str = "outputs"):
        """Save Grounding DINO detection results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save JSON results
        json_path = f"{output_dir}/dino_{base_name}_detections.json"
        analysis = self.analyze_detections(detections)
        
        result_data = {
            "model": "Grounding DINO",
            "image": image_path,
            "detections": detections,
            "analysis": analysis,
            "model_info": {
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "confidence_threshold": self.confidence_threshold
            }
        }
        
        with open(json_path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f" Grounding DINO results saved: {json_path}")
        
        # Save visualization
        vis_path = f"{output_dir}/dino_{base_name}_visualization.jpg"
        self.create_visualization(image_path, detections, vis_path)
        
        return json_path, vis_path


def grounding_dino(image_path, reference_json_path, use_gpu=True, create_overlay=True):
    """
    Main Grounding DINO function for object detection and confusion matrix analysis
    
    Args:
        image_path: Path to the image file to analyze
        reference_json_path: Path to ground truth JSON file
        use_gpu: Whether to use GPU if available (default: True)
        create_overlay: Whether to create overlay visualization (default: True)
    
    Returns:
        dict: Formatted confusion matrix results (includes 'overlay_path' if created)
    """
    print("Running Grounding DINO model...")
    
    # Load reference data
    with open(reference_json_path, 'r') as f:
        reference_data = json.load(f)
    
    # Handle different JSON formats
    # Format 1: [{"class": "...", "bbox": [...]}, ...]
    # Format 2: {"image": "...", "objects": [{"class": "...", "bbox": [...]}]}
    if isinstance(reference_data, dict) and 'objects' in reference_data:
        reference_data = reference_data['objects']
    
    # Detect objects in the image
    detected_objects = detect_objects_in_image(image_path, use_gpu)
    
    # Create confusion matrix and process detections
    confusion_matrix = ConfusionMatrix(reference_data)
    
    # Process each detected object through the confusion matrix
    for obj in detected_objects:
        confusion_matrix.handle_object_data(obj['class'], obj['bbox'])
    
    # Get metrics and format results
    class_metrics, mean_ap, mean_f1, mean_accuracy = confusion_matrix.get_matrix_metrics()
    
    # Format and print results
    results = format_results(class_metrics, mean_ap, mean_f1, mean_accuracy, confusion_matrix)
    print_results(results)
    
    return results


def detect_objects_in_image(image_path, use_gpu=True):
    """
    Detect objects in an image using Grounding DINO model
    
    Args:
        image_path: Path to the image file
        use_gpu: Whether to use GPU if available
    
    Returns:
        list: List of detected objects with format [{'class': str, 'bbox': [x1,y1,x2,y2]}, ...]
    """
    try:
        # Create detector
        device = "auto" if use_gpu else "cpu"
        detector = GroundingDINODetector(device=device)
        
        if detector.model is None:
            print("[ERROR] Grounding DINO model not loaded")
            return []
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return []
        
        print(f" Running Grounding DINO on {image_path}")
        
        # Load image
        from groundingdino.util.inference import load_image, predict
        image_source, image = load_image(image_path)
        h, w, _ = image_source.shape
        
        # Use INDOOR_BUSINESS_CLASSES for prompts
        text_query = ". ".join(INDOOR_BUSINESS_CLASSES) + "."
        
        # Run detection
        boxes, confidences, labels = predict(
            model=detector.model,
            image=image,
            caption=text_query,
            box_threshold=detector.box_threshold,
            text_threshold=detector.text_threshold,
            device="cpu"
        )
        
        # Convert coordinates: [cx, cy, w, h] -> [x1, y1, x2, y2]
        detected_objects = []
        for box, confidence, label in zip(boxes, confidences, labels):
            if confidence >= detector.confidence_threshold:
                cx_norm, cy_norm, w_norm, h_norm = box
                
                cx = cx_norm * w
                cy = cy_norm * h
                box_w = w_norm * w
                box_h = h_norm * h
                
                x1 = int(cx - box_w / 2)
                y1 = int(cy - box_h / 2)
                x2 = int(cx + box_w / 2)
                y2 = int(cy + box_h / 2)
                
                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Validate and filter to only INDOOR_BUSINESS_CLASSES
                if x2 > x1 and y2 > y1:
                    class_name = label.strip().lower()
                    # Only include objects that are in INDOOR_BUSINESS_CLASSES
                    if class_name in INDOOR_BUSINESS_CLASSES:
                        detected_objects.append({
                            'class': class_name,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence)
                        })
        
        print(f"Detected {len(detected_objects)} relevant objects")
        return detected_objects
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return []


def format_results(class_metrics, mean_ap, mean_f1, mean_accuracy, confusion_matrix):
    """
    Format confusion matrix results for display
    
    Args:
        class_metrics: List of ObjectMetrics objects
        mean_ap: Mean average precision
        mean_f1: Mean F1 score
        confusion_matrix: ConfusionMatrix object
    
    Returns:
        dict: Formatted results
    """
    results = {
        'mean_average_precision': mean_ap,
        'mean_f1_score': mean_f1,
        'class_metrics': {},
        'confusion_matrix': confusion_matrix.get_confusion_matrix().tolist(),
        'unmatched_objects': len(confusion_matrix.unmatched_objects),
        'missing_objects': len(confusion_matrix.missing_objects)
    }
    
    # Format class-specific metrics
    for metric in class_metrics:
        results['class_metrics'][metric.class_name] = {
            'precision': metric.precision,
            'sensitivity': metric.sensitivity,
            'specificity': metric.specificity,
            'f1_score': metric.f1_score
        }
    
    return results


def print_results(results):
    """
    Print formatted confusion matrix results
    
    Args:
        results: Formatted results dictionary
    """
    print("\n" + "="*60)
    print("GROUNDING DINO CONFUSION MATRIX RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Mean Average Precision: {results['mean_average_precision']:.4f}")
    print(f"  Mean F1 Score: {results['mean_f1_score']:.4f}")
    print(f"  Unmatched Objects: {results['unmatched_objects']}")
    print(f"  Missing Objects: {results['missing_objects']}")
    
    print(f"\nClass-Specific Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1 Score':<10}")
    print("-" * 60)
    
    for class_name, metrics in results['class_metrics'].items():
        print(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['sensitivity']:<12.4f} "
              f"{metrics['specificity']:<12.4f} {metrics['f1_score']:<10.4f}")
    
    print("\n" + "="*60)


def test_all_photos(test_photos_dir: str = "test_photos", use_gpu=True):
    """
    Test Grounding DINO on all images in the test_photos directory
    
    Args:
        test_photos_dir: Directory containing images/ and labels/ subdirectories
        use_gpu: Whether to use GPU if available
    
    Returns:
        dict: Summary results for all tested images
    """
    import time
    start_time = time.time()
    
    print("="*60)
    print("TESTING GROUNDING DINO ON ALL TEST PHOTOS")
    print("="*60)
    
    images_dir = os.path.join(test_photos_dir, "images")
    labels_dir = os.path.join(test_photos_dir, "labels")
    
    if not os.path.exists(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        return {}
    
    if not os.path.exists(labels_dir):
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return {}
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(image_extensions)]
    image_files.sort()
    
    if not image_files:
        print(f"[ERROR] No image files found in: {images_dir}")
        return {}
    
    print(f"\nFound {len(image_files)} test images")
    print("-"*60)
    
    # Process each image
    all_results = {}
    successful_tests = 0
    failed_tests = 0
    
    for i, image_file in enumerate(image_files, 1):
        # Construct paths
        image_path = os.path.join(images_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        label_file = f"{base_name}.json"
        label_path = os.path.join(labels_dir, label_file)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"\n[{i}/{len(image_files)}] [WARNING]  Skipping {image_file} - no matching label file")
            failed_tests += 1
            continue
        
        print(f"\n[{i}/{len(image_files)}]  Testing: {image_file}")
        print(f"              Label: {label_file}")
        
        try:
            # Run Grounding DINO with confusion matrix analysis
            results = grounding_dino(image_path, label_path, use_gpu=use_gpu, create_overlay=True)
            
            # Store results
            all_results[image_file] = {
                'results': results,
                'image_path': image_path,
                'label_path': label_path,
                'status': 'success'
            }
            
            # Print summary for this image
            print(f"   [OK] mAP: {results['mean_average_precision']:.4f}, "
                  f"mF1: {results['mean_f1_score']:.4f}, "
                  f"Unmatched: {results['unmatched_objects']}, "
                  f"Missing: {results['missing_objects']}")
            
            successful_tests += 1
            
        except Exception as e:
            print(f"   [ERROR] Failed: {str(e)}")
            all_results[image_file] = {
                'error': str(e),
                'image_path': image_path,
                'label_path': label_path,
                'status': 'failed'
            }
            failed_tests += 1
    
    # Print overall summary
    print("\n" + "="*60)
    print(" OVERALL TEST SUMMARY")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    
    # Calculate average metrics across successful tests
    if successful_tests > 0:
        avg_map = sum(r['results']['mean_average_precision'] 
                     for r in all_results.values() 
                     if r['status'] == 'success') / successful_tests
        avg_f1 = sum(r['results']['mean_f1_score'] 
                    for r in all_results.values() 
                    if r['status'] == 'success') / successful_tests
        
        print(f"\nAverage Metrics Across All Tests:")
        print(f"  Average mAP: {avg_map:.4f}")
        print(f"  Average mF1: {avg_f1:.4f}")
        
        # Show per-image results
        print(f"\nPer-Image Results:")
        print("-"*60)
        for image_file, result_data in all_results.items():
            if result_data['status'] == 'success':
                res = result_data['results']
                print(f"{image_file:20s} | mAP: {res['mean_average_precision']:.4f} | "
                      f"mF1: {res['mean_f1_score']:.4f} | "
                      f"Unmatched: {res['unmatched_objects']:2d} | "
                      f"Missing: {res['missing_objects']:2d}")
    
    print("="*60)
    
    # Display total execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Total Execution Time per image: {elapsed_time/len(image_files):.2f} seconds")
    print("="*60)
    
    return all_results


def dino_only(image1_path: str = "IMG_1464.jpg", image2_path: str = "IMG_1465.jpg"):
    """Grounding DINO-only detection for timing comparison"""
    print("Running Grounding DINO-only detection...")
    
    detector = GroundingDINODetector()
    if detector.model is None:
        return "Grounding DINO not available"
    
    total_objects = 0
    for img_path in [image1_path, image2_path]:
        if os.path.exists(img_path):
            detections = detector.detect_objects(img_path)
            total_objects += len(detections)
    
    return f"Grounding DINO completed - {total_objects} objects detected"


def comprehensive_dino_analysis(image1_path: str = "IMG_1464.jpg", image2_path: str = "IMG_1465.jpg"):
    """Comprehensive Grounding DINO analysis and comparison"""
    print("=" * 60)
    print(" GROUNDING DINO COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    detector = GroundingDINODetector()
    
    if detector.model is None:
        print("[ERROR] Cannot run - Grounding DINO not available")
        return
    
    # Check if images exist
    for img_path in [image1_path, image2_path]:
        if not os.path.exists(img_path):
            print(f"[ERROR] {img_path} not found")
            return
    
    results = {}
    
    # Analyze both images
    for img_path in [image1_path, image2_path]:
        print(f"\n Analyzing {img_path}")
        print("-" * 40)
        
        detections = detector.detect_objects(img_path)
        
        # Analyze objects by category
        objects = [det["object"] for det in detections]
        
        # Define categories
        categories = {
            "furniture": ["chair", "table", "desk", "shelf", "cabinet", "furniture", "bookshelf"],
            "technology": ["computer", "laptop", "monitor", "screen", "keyboard", "mouse", "printer", "phone"],
            "supplies": ["book", "paper", "document", "folder", "pen", "pencil", "notebook"],
            "storage": ["box", "bag", "backpack", "basket", "bin", "container", "storage"],
            "lighting": ["lamp", "light", "bulb"],
            "decorative": ["plant", "flower", "picture", "painting", "frame", "clock"],
            "personal": ["bottle", "cup", "mug"],
            "room_elements": ["window", "door", "wall", "floor"]
        }
        
        # Count objects by category
        category_counts = {}
        for category, keywords in categories.items():
            count = sum(1 for obj in objects if any(keyword in obj for keyword in keywords))
            if count > 0:
                category_counts[category] = count
        
        # Save results with DINO-specific naming
        json_path, vis_path = detector.save_results(img_path, detections)
        
        print(f"[OK] Objects detected: {len(detections)}")
        print(f" Results saved: {json_path}")
        print(f"  Visualization: {vis_path}")
        print(f" Categories: {category_counts}")
        
        # Show top detections
        if detections:
            print("Top detections:")
            sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            for i, det in enumerate(sorted_detections[:8], 1):
                print(f"  {i}. {det['object']}: {det['confidence']:.3f}")
        
        results[img_path] = {
            "detections": detections,
            "categories": category_counts,
            "total": len(detections)
        }
    
    # Compare images
    if len(results) == 2:
        print(f"\n{'='*60}")
        print(" COMPARISON")
        print('='*60)
        
        img1, img2 = list(results.keys())
        objects1 = set(det["object"] for det in results[img1]["detections"])
        objects2 = set(det["object"] for det in results[img2]["detections"])
        
        common = objects1 & objects2
        only_img1 = objects1 - objects2
        only_img2 = objects2 - objects1
        
        print(f"Common objects: {len(common)}")
        print(f"Only in {img1}: {sorted(only_img1)}")
        print(f"Only in {img2}: {sorted(only_img2)}")
        print(f"Total differences: {len(only_img1) + len(only_img2)}")
    
    print(f"\n{'='*60}")
    print(" GROUNDING DINO SUMMARY")  
    print(f"{'='*60}")
    print(f" Results saved to: outputs/dino_*")
    print(f" Analysis complete!")


def test_confusion_matrix(image_path: str = "test_photos/images/bedroom1.jpeg"):
    """Easy test function for confusion matrix - just call this!"""
    print(f" TESTING GROUNDING DINO CONFUSION MATRIX")
    print(f"Image: {image_path}")

    try:
        detector = GroundingDINODetector()
        detections = detector.detect_objects(image_path)

        if detections:
            print(f"[OK] SUCCESS: Detected {len(detections)} objects")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            vis_path = f"outputs/dino_{base_name}_confusion_matrix.jpg"
            print(f"  Visualization saved: {vis_path}")
            return True
        else:
            print("[ERROR] No objects detected")
            return False

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Check if user wants to test all photos
    if len(sys.argv) > 1 and ("--all" in sys.argv) and ("--no-gpu" in sys.argv):
        # Test all photos in test_photos directory
        test_all_photos(test_photos_dir="test_photos", use_gpu=False)
    elif len(sys.argv) > 1 and ("--all" in sys.argv):
        test_all_photos(test_photos_dir="test_photos", use_gpu=True)
    else:
        # Test single image (default behavior matching yolox.py interface)
        print("="*60)
        print("Testing Grounding DINO with model_tests.py interface")
        print("="*60)
        print("Tip: Run with '--all' to test all images in test_photos/")
        print("="*60)
        
        # Example usage matching yolox.py
        image_path = "test_photos/images/bedroom1.jpeg"
        reference_json_path = "test_photos/labels/bedroom1.json"
        
        # Run Grounding DINO with confusion matrix analysis
        results = grounding_dino(image_path, reference_json_path, use_gpu=True, create_overlay=True)
        
        print("\n" + "="*60)
        print("Test Complete!")
        print("="*60)
        print("\nTo test all images, run: python grounding_dino.py --all")
        print("="*60)
