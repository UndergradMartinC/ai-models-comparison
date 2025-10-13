# Import required libraries
import torch                                                   # PyTorch for deep learning
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor  # Hugging Face RF-DETR model
from PIL import Image                                          # Image loading and processing
import json                                                    # For JSON handling
from COCO_CLASSES import INDOOR_BUSINESS_CLASSES, CONFIDENCE_THRESHOLD
from model_tests import ConfusionMatrix

def rfdetr(image_path, reference_json_path, use_gpu=True):
    """
    Main RF-DETR function for object detection and confusion matrix analysis
    
    Args:
        image_path: Path to the image file to analyze
        reference_json_path: Path to ground truth JSON file
        use_gpu: Whether to use GPU if available (default: True)
    
    Returns:
        dict: Formatted confusion matrix results
    """
    print("Running RF-DETR model...")
    
    # Load reference data
    with open(reference_json_path, 'r') as f:
        reference_data = json.load(f)
    
    # Detect objects in the image
    detected_objects = detect_objects_in_image(image_path, use_gpu)
    
    # Create confusion matrix and process detections
    confusion_matrix = ConfusionMatrix(reference_data)
    
    # Process each detected object through the confusion matrix
    for obj in detected_objects:
        confusion_matrix.handle_object_data(obj['class'], obj['bbox'], obj['class'])
    
    # Get metrics and format results
    class_metrics, mean_ap, mean_f1 = confusion_matrix.get_matrix_metrics()
    
    # Format and print results
    results = format_results(class_metrics, mean_ap, mean_f1, confusion_matrix)
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


# Main execution example
if __name__ == "__main__":
    # Example usage
    image_path = "before.jpg"
    reference_json_path = "sample_ground_truth.json"
    
    # Run RF-DETR with GPU (default)
    results = rfdetr(image_path, reference_json_path, use_gpu=True)
    
    # Or run with CPU only
    # results = rfdetr(image_path, reference_json_path, use_gpu=False)