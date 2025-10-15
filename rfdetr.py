# Import required libraries
import torch                                                   # PyTorch for deep learning
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor  # Hugging Face RF-DETR model
from PIL import Image, ImageDraw, ImageFont                    # Image loading and processing
import json                                                    # For JSON handling
from COCO_CLASSES import INDOOR_BUSINESS_CLASSES, CONFIDENCE_THRESHOLD
from model_tests import ConfusionMatrix

def rfdetr(image_path, reference_json_path, use_gpu=True, create_overlay=True):
    """
    Main RF-DETR function for object detection and confusion matrix analysis
    
    Args:
        image_path: Path to the image file to analyze
        reference_json_path: Path to ground truth JSON file
        use_gpu: Whether to use GPU if available (default: True)
        create_overlay: Whether to create overlay visualization (default: True)
    
    Returns:
        dict: Formatted confusion matrix results (includes 'overlay_path' if created)
    """
    print("Running RF-DETR model...")
    
    # Load reference data
    with open(reference_json_path, 'r') as f:
        reference_data = json.load(f)
    
    # Detect objects in the image
    detected_objects = detect_objects_in_image(image_path, use_gpu)
    """
    # Create overlay visualization if requested
    overlay_path = None
    if create_overlay:
        import os
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        overlay_filename = f"rfdetr_overlay_{base_name}.jpg"
        overlay_path = create_bounding_box_overlay(
            image_path, 
            reference_json_path, 
            detected_objects,
            overlay_filename
        )
    """
    
    # Create confusion matrix and process detections
    confusion_matrix = ConfusionMatrix(reference_data)
    
    # Process each detected object through the confusion matrix
    for obj in detected_objects:
        confusion_matrix.handle_object_data(obj['class'], obj['bbox'])
    
    # Get metrics and format results
    class_metrics, mean_ap, mean_f1 = confusion_matrix.get_matrix_metrics()
    
    # Format and print results
    results = format_results(class_metrics, mean_ap, mean_f1, confusion_matrix)
    """
    # Add overlay path to results
    if overlay_path:
        results['overlay_path'] = overlay_path
        print(f"\n Overlay visualization saved: {overlay_path}")
    """
    print_results(results)
    
    return results


def detect_objects_in_image(image_path, use_gpu=True):
    """
    Detect objects in an image using RF-DETR model
    
    Args:
        image_path: Path to the image file
        use_gpu: Whether to use GPU if available
    
    Returns:
        list: List of detected objects with format [{'class': str, 'bbox': [x1,y1,x2,y2]}, ...]
    """
    try:
        # Setup device
        device = setup_device(use_gpu)
        
        # Load model and processor
        print("Loading RF-DETR model...")
        model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
        processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        model = model.to(device)
        model.eval()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, 
            threshold=CONFIDENCE_THRESHOLD, 
            target_sizes=target_sizes
        )[0]
        
        # Convert to required format and filter for indoor business classes
        detected_objects = []
        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        # Create COCO class mapping
        coco_to_indoor = get_coco_class_mapping()
        
        for box, label, score in zip(boxes, labels, scores):
            if label in coco_to_indoor:
                class_name = coco_to_indoor[label]
                x1, y1, x2, y2 = box.astype(int)
                detected_objects.append({
                    'class': class_name,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score)
                })
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Detected {len(detected_objects)} relevant objects")
        return detected_objects
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return []


def setup_device(use_gpu=True):
    """
    Setup PyTorch device (GPU or CPU)
    
    Args:
        use_gpu: Whether to use GPU if available
    
    Returns:
        torch.device: Device to use for computation
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def get_coco_class_mapping():
    """
    Create mapping from COCO class indices to indoor business class names
    
    Returns:
        dict: Mapping from COCO index to class name
    """
    # COCO class indices for indoor business objects
    coco_mapping = {
            0: "person",           # People in the space
            24: "backpack",        # Luggage/personal items
            26: "handbag",         # Personal bags
            27: "tie",             # Professional attire
            28: "suitcase",        # Travel luggage (Airbnb guests)
            39: "bottle",          # Water bottles, beverages
            40: "wine glass",      # Dining/entertainment
            41: "cup",             # Coffee cups, mugs
            42: "fork",            # Dining utensils
            43: "knife",           # Kitchen utensils
            44: "spoon",           # Dining utensils
            45: "bowl",            # Dishes
            46: "banana",          # Food items
            47: "apple",           # Food items
            48: "sandwich",        # Food items
            56: "chair",           # Office/dining furniture
            57: "couch",           # Lounge furniture
            58: "potted plant",    # Decor/ambiance
            59: "bed",             # Bedroom furniture
            60: "dining table",    # Dining/meeting furniture
            61: "toilet",          # Bathroom fixtures
            62: "tv",              # Entertainment/presentation
            63: "laptop",          # Work equipment
            64: "mouse",           # Computer peripherals
            65: "remote",          # TV/AC controls
            66: "keyboard",        # Computer equipment
            67: "cell phone",      # Personal devices
            68: "microwave",       # Kitchen appliances
            69: "oven",            # Kitchen appliances
            70: "toaster",         # Kitchen appliances
            71: "sink",            # Kitchen/bathroom fixtures
            72: "refrigerator",    # Kitchen appliances
            73: "book",            # Reading materials/decor
            74: "clock",           # Time displays
            75: "vase",            # Decorative items
            76: "scissors",        # Office supplies
            77: "teddy bear",      # Comfort items (hotels)
            78: "hair drier",      # Bathroom amenities
            79: "toothbrush"       # Personal hygiene items
        }
    return coco_mapping


def format_results(class_metrics, mean_ap, mean_f1, confusion_matrix):
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
    print("RF-DETR CONFUSION MATRIX RESULTS")
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


def create_bounding_box_overlay(image_path, reference_json_path, detected_objects, output_path="overlay_result.jpg"):
    """
    Create an overlay visualization with bounding boxes
    
    Args:
        image_path: Path to the original image
        reference_json_path: Path to ground truth JSON file
        detected_objects: List of detected objects from RF-DETR
        output_path: Path to save the overlay image
    
    Returns:
        str: Path to the saved overlay image
    """
    print(f"Creating overlay visualization...")
    print(f"  Image: {image_path}")
    print(f"  Reference: {reference_json_path}")
    print(f"  Detected objects: {len(detected_objects)}")
    print(f"  Output: {output_path}")
    
    try:
        # Check if image file exists
        import os
        if not os.path.exists(image_path):
            print(f"ERROR: Image file not found: {image_path}")
            return None
            
        # Load the original image
        print("Loading image...")
        image = Image.open(image_path).convert("RGB")
        print(f"Image size: {image.size}")
        draw = ImageDraw.Draw(image)
        
        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                # Try other common font paths
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
            except:
                font = ImageFont.load_default()
                print("Using default font")
        
        # Load reference data
        print("Loading reference data...")
        if os.path.exists(reference_json_path):
            with open(reference_json_path, 'r') as f:
                reference_data = json.load(f)
            print(f"Reference objects: {len(reference_data)}")
        else:
            print(f"WARNING: Reference file not found: {reference_json_path}")
            reference_data = []
        
        # Draw reference objects (BLUE boxes)
        print("Drawing reference objects (blue)...")
        for i, ref_obj in enumerate(reference_data):
            x1, y1, x2, y2 = ref_obj['bbox']
            class_name = ref_obj['class']
            print(f"  Drawing reference {i+1}: {class_name} at [{x1}, {y1}, {x2}, {y2}]")
            
            # Draw blue bounding box for reference
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            
            # Add blue label (handle text positioning better)
            label = f"REF: {class_name}"
            label_y = max(y1-25, 5)  # Ensure label stays within image bounds
            try:
                bbox = draw.textbbox((x1, label_y), label, font=font)
                draw.rectangle(bbox, fill="blue")
                draw.text((x1, label_y), label, fill="white", font=font)
            except:
                # Fallback if textbbox not available (older PIL versions)
                draw.text((x1, label_y), label, fill="blue", font=font)
        
        # Draw detected objects (RED boxes)
        print("Drawing detected objects (red)...")
        for i, det_obj in enumerate(detected_objects):
            x1, y1, x2, y2 = det_obj['bbox']
            class_name = det_obj['class']
            confidence = det_obj.get('confidence', 0.0)
            print(f"  Drawing detection {i+1}: {class_name} at [{x1}, {y1}, {x2}, {y2}] (conf: {confidence:.2f})")
            
            # Draw red bounding box for detection
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Add red label with confidence (handle text positioning better)
            label = f"DET: {class_name} ({confidence:.2f})"
            label_y = min(y2+5, image.size[1]-25)  # Ensure label stays within image bounds
            try:
                bbox = draw.textbbox((x1, label_y), label, font=font)
                draw.rectangle(bbox, fill="red")
                draw.text((x1, label_y), label, fill="white", font=font)
            except:
                # Fallback if textbbox not available (older PIL versions)
                draw.text((x1, label_y), label, fill="red", font=font)
        
        # Add legend
        legend_x, legend_y = 10, 10
        draw.rectangle([legend_x, legend_y, legend_x+250, legend_y+50], fill="white", outline="black", width=2)
        draw.text((legend_x+10, legend_y+10), "BLUE: Reference (Ground Truth)", fill="blue", font=font)
        draw.text((legend_x+10, legend_y+25), "RED: Detected Objects", fill="red", font=font)
        
        # Save the overlay image
        print(f"Saving overlay to: {output_path}")
        image.save(output_path, "JPEG", quality=95)
        
        # Verify file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Overlay saved successfully: {output_path} ({file_size} bytes)")
            return output_path
        else:
            print(f"✗ Failed to save overlay: {output_path}")
            return None
        
    except Exception as e:
        print(f"ERROR creating overlay: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def rfdetr_with_overlay(image_path, reference_json_path, use_gpu=True, save_overlay=True):
    """
    Run RF-DETR detection and create overlay visualization
    
    Args:
        image_path: Path to the image file to analyze
        reference_json_path: Path to ground truth JSON file
        use_gpu: Whether to use GPU if available (default: True)
        save_overlay: Whether to save the overlay visualization (default: True)
    
    Returns:
        tuple: (confusion_matrix_results, overlay_path)
    """
    print("Running RF-DETR with overlay visualization...")
    
    # Load reference data
    with open(reference_json_path, 'r') as f:
        reference_data = json.load(f)
    
    # Detect objects in the image
    detected_objects = detect_objects_in_image(image_path, use_gpu)
    
    # Create overlay visualization if requested
    overlay_path = None
    if save_overlay:
        overlay_path = create_bounding_box_overlay(
            image_path, 
            reference_json_path, 
            detected_objects,
            f"overlay_{image_path.split('/')[-1].split('.')[0]}.jpg"
        )
    
    # Create confusion matrix and process detections
    confusion_matrix = ConfusionMatrix(reference_data)
    
    # Process each detected object through the confusion matrix
    for obj in detected_objects:
        confusion_matrix.handle_object_data(obj['class'], obj['bbox'])
    
    # Get metrics and format results
    class_metrics, mean_ap, mean_f1 = confusion_matrix.get_matrix_metrics()
    
    # Format and print results
    results = format_results(class_metrics, mean_ap, mean_f1, confusion_matrix)
    print_results(results)
    
    return results, overlay_path


def test_confusion_matrix():
    """
    Dummy function to test confusion matrix with simulated detections
    Tests against sample_ground_truth.json which contains:
    - person at [100, 100, 200, 300]
    - chair at [50, 150, 120, 250] 
    - laptop at [300, 200, 450, 300]
    - cup at [200, 50, 230, 100]
    """
    print("Testing Confusion Matrix with Dummy Data...")
    print("="*60)
    
    # Load ground truth
    reference_json_path = "sample_ground_truth.json"
    with open(reference_json_path, 'r') as f:
        reference_data = json.load(f)
    
    print(f"Ground Truth Objects: {len(reference_data)}")
    for obj in reference_data:
        print(f"  - {obj['class']} at {obj['bbox']}")
    
    # Create confusion matrix
    confusion_matrix = ConfusionMatrix(reference_data)
    
    # Simulate detected objects - mix of correct matches, misclassifications, and false positives
    simulated_detections = [
        # TRUE POSITIVES (close matches to ground truth)
        {'class': 'person', 'bbox': [105, 105, 195, 295], 'confidence': 0.85},    # Close to person
        {'class': 'chair', 'bbox': [55, 155, 115, 245], 'confidence': 0.78},     # Close to chair
        {'class': 'laptop', 'bbox': [305, 205, 445, 295], 'confidence': 0.92},   # Close to laptop
        
        # MISCLASSIFICATION (correct location, wrong class)
        {'class': 'bottle', 'bbox': [205, 55, 225, 95], 'confidence': 0.65},     # Cup detected as bottle
        
        # FALSE POSITIVES (objects not in ground truth)
        {'class': 'book', 'bbox': [400, 100, 450, 150], 'confidence': 0.70},     # New object
        {'class': 'mouse', 'bbox': [500, 300, 520, 320], 'confidence': 0.60},    # New object
        
        # DUPLICATE DETECTION (same object detected twice)
        {'class': 'person', 'bbox': [110, 110, 190, 290], 'confidence': 0.75},   # Another person detection
    ]
    
    print(f"\nSimulated Detections: {len(simulated_detections)}")
    for i, obj in enumerate(simulated_detections, 1):
        print(f"  {i}. {obj['class']} at {obj['bbox']} (conf: {obj['confidence']:.2f})")
    
    # Process each detected object through confusion matrix
    print(f"\nProcessing detections through confusion matrix...")
    for obj in simulated_detections:
        confusion_matrix.handle_object_data(obj['class'], obj['bbox'], obj['class'])
        print(f"  Processed: {obj['class']} at {obj['bbox']}")
    
    # Get metrics and format results
    class_metrics, mean_ap, mean_f1 = confusion_matrix.get_matrix_metrics()
    
    # Print key results
    print(f"\n" + "="*60)
    print("CONFUSION MATRIX TEST RESULTS")
    print("="*60)
    print(f"Mean Average Precision: {mean_ap:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"Unmatched Objects (False Positives): {len(confusion_matrix.unmatched_objects)}")
    print(f"Missing Objects (False Negatives): {len(confusion_matrix.missing_objects)}")
    
    # Show unmatched objects
    if confusion_matrix.unmatched_objects:
        print(f"\nUnmatched Objects:")
        for obj in confusion_matrix.unmatched_objects:
            print(f"  - {obj.class_name} at {obj.bbox}")
    
    # Show missing objects  
    if confusion_matrix.missing_objects:
        print(f"\nMissing Objects:")
        for obj in confusion_matrix.missing_objects:
            print(f"  - {obj.class_name} at {obj.bbox}")
    
    print("="*60)
    
    return mean_ap, mean_f1


def test_overlay_with_dummy_data():
    """
    Test the overlay function with dummy detected objects
    """
    print("Testing overlay with dummy data...")
    
    # Simulate some detected objects
    dummy_detections = [
        {'class': 'person', 'bbox': [105, 105, 195, 295], 'confidence': 0.85},
        {'class': 'chair', 'bbox': [55, 155, 115, 245], 'confidence': 0.78},
        {'class': 'laptop', 'bbox': [305, 205, 445, 295], 'confidence': 0.92},
        {'class': 'bottle', 'bbox': [205, 55, 225, 95], 'confidence': 0.65},
    ]
    
    # Test with available files
    image_path = "before.jpg"
    reference_json_path = "sample_ground_truth.json"
    
    # Check if files exist
    import os
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        print("Available image files:")
        for file in os.listdir("."):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"  - {file}")
        return None
    
    if not os.path.exists(reference_json_path):
        print(f"Reference file not found: {reference_json_path}")
        return None
    
    # Create overlay
    overlay_path = create_bounding_box_overlay(
        image_path, 
        reference_json_path, 
        dummy_detections,
        "test_overlay.jpg"
    )
    
    return overlay_path


# Main execution example
if __name__ == "__main__":
    # Test confusion matrix with dummy data
    test_confusion_matrix()
    
    # Test overlay with dummy data
    print("\n" + "="*60)
    print("Testing overlay visualization...")
    print("="*60)
    overlay_path = test_overlay_with_dummy_data()
    if overlay_path:
        print(f"✓ Test overlay created: {overlay_path}")
    else:
        print("✗ Failed to create test overlay")
    
    print("\n" + "="*60)
    print("To run actual RF-DETR detection, uncomment the lines below:")
    print("="*60)
    
    # # Example usage with real RF-DETR detection
    # image_path = "before.jpg"
    # reference_json_path = "sample_ground_truth.json"
    # 
    # # Run RF-DETR with overlay visualization (default)
    # results = rfdetr(image_path, reference_json_path, use_gpu=True, create_overlay=True)
    # 
    # # Or run without overlay
    # # results = rfdetr(image_path, reference_json_path, use_gpu=True, create_overlay=False)
    # 
    # # Or use the separate overlay function
    # # results, overlay_path = rfdetr_with_overlay(image_path, reference_json_path, use_gpu=True)