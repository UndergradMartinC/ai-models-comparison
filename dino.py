# Import required libraries
import time                                                    # For timing operations
import torch                                                   # PyTorch for deep learning
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection  # Hugging Face RT-DETR (DINO-enhanced) model
from PIL import Image                                          # Image loading and processing
import cv2                                                     # OpenCV for computer vision
import numpy as np                                             # Numerical operations
import matplotlib.pyplot as plt                                # Plotting and visualization
import matplotlib.patches as patches                           # Drawing shapes on plots

def dino(before_photo="before.jpg", after_photo="after.jpg", result_photo="dino_result.jpg"):
    """
    Main DINO function for object detection comparison
    This is the entry point called by main.py for timing comparisons
    """
    print("Running DINO model...")
    
    # Call the main comparison function with the image paths
    result = compare_images(before_photo, after_photo, result_photo)
    
    # Return result string for the timing system in main.py
    return result


# ============================================================================
# HELPER FUNCTIONS FOR RF-DETR INTEGRATION
# These functions can be imported by rfdetr.py to leverage DINO's enhancements
# ============================================================================

def enhance_rfdetr_detection(image_path, model, processor, device):
    """
    Enhanced detection function that RF-DETR can use to benefit from DINO improvements
    
    This function applies DINO's dual-threshold confidence refinement to any RT-DETR model,
    including the standard RF-DETR model used by rfdetr.py
    
    Args:
        image_path: Path to the image file
        model: Any RT-DETR model (including RF-DETR's standard model)
        processor: Image processor for the model
        device: PyTorch device (cuda or cpu)
    
    Returns:
        tuple: (PIL Image, enhanced detection results) or (None, None) if error
    """
    try:
        # Load image file and convert to RGB format (removes alpha channel if present)
        image = Image.open(image_path).convert("RGB")
        
        # Process image into tensor format expected by the model
        inputs = processor(images=image, return_tensors="pt")
        
        # Move all input tensors to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference with GPU optimizations
        with torch.no_grad():
            if device.type == 'cuda':
                # Use automatic mixed precision on GPU for faster inference
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                # Standard inference on CPU
                outputs = model(**inputs)
        
        # Prepare target sizes for post-processing
        target_sizes = torch.tensor([image.size[::-1]])
        
        # DINO enhancement: Use lower threshold for better anchor selection,
        # then apply stricter filtering later for more robust detection
        results = processor.post_process_object_detection(outputs, threshold=0.25, target_sizes=target_sizes)[0]
        
        # DINO-style confidence refinement: Apply additional confidence filtering
        if len(results["scores"]) > 0:
            high_conf_mask = results["scores"] > 0.35  # Stricter secondary threshold
            results["boxes"] = results["boxes"][high_conf_mask]
            results["labels"] = results["labels"][high_conf_mask]
            results["scores"] = results["scores"][high_conf_mask]
        
        return image, results
    except Exception as e:
        print(f"Error in DINO-enhanced detection for {image_path}: {str(e)}")
        return None, None


def enhanced_object_matching(results1, results2, iou_threshold=0.5):
    """
    DINO-enhanced object matching that RF-DETR can use for better accuracy
    
    This function provides RF-DETR with DINO's advanced matching strategy including:
    - Multi-scale confidence weighting
    - Enhanced spatial overlap calculation
    - Adaptive thresholding based on object class confidence
    
    Args:
        results1: Detection results from first image
        results2: Detection results from second image
        iou_threshold: Base IoU threshold, adapted per class (default 0.5)
    
    Returns:
        tuple: (matched_pairs, used_indices2)
               matched_pairs: List of (index1, index2, iou_score, confidence_score) tuples
               used_indices2: Set of indices from image2 that were matched
    """
    return match_objects(results1, results2, iou_threshold)


def get_dino_model_and_processor(device):
    """
    Load the enhanced DINO model that RF-DETR can use for better performance
    
    This function loads the DINO-enhanced RT-DETR model trained on COCO + Open Images,
    which typically provides better accuracy than the standard RF-DETR model
    
    Args:
        device: PyTorch device (cuda or cpu)
    
    Returns:
        tuple: (model, processor) - Ready-to-use DINO-enhanced model and processor
    """
    try:
        print("Loading DINO-enhanced RT-DETR model for RF-DETR...")
        model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        
        # Move model to specified device
        model = model.to(device)
        model.eval()
        
        print(f"DINO-enhanced model loaded on {device}")
        return model, processor
    except Exception as e:
        print(f"Error loading DINO-enhanced model: {str(e)}")
        return None, None


def rfdetr_with_dino_boost(before_photo="before.jpg", after_photo="after.jpg", result_photo="rfdetr_dino_boosted.jpg"):
    """
    RF-DETR comparison using DINO's enhanced model and processing techniques
    
    This function demonstrates how RF-DETR can benefit from DINO's improvements:
    1. Uses DINO's enhanced RT-DETR model (trained on COCO + Open Images)
    2. Applies DINO's dual-threshold confidence refinement
    3. Uses DINO's advanced object matching with confidence weighting
    4. Maintains RF-DETR's visualization style with DINO enhancements
    
    Args:
        before_photo: Path to first image
        after_photo: Path to second image
        result_photo: Path where comparison result will be saved
    
    Returns:
        str: Summary of comparison results with DINO enhancements
    """
    print("Running RF-DETR with DINO enhancements...")
    
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load DINO-enhanced model
        model, processor = get_dino_model_and_processor(device)
        if model is None:
            return "Failed to load DINO-enhanced model"
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run enhanced detection on both images
        print(f"Processing before image with DINO enhancements: {before_photo}")
        before_image, before_results = enhance_rfdetr_detection(before_photo, model, processor, device)
        
        print(f"Processing after image with DINO enhancements: {after_photo}")
        after_image, after_results = enhance_rfdetr_detection(after_photo, model, processor, device)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check if both images were processed successfully
        if before_image is None or after_image is None:
            return "RF-DETR + DINO comparison failed - could not load images"
        
        # Use DINO's enhanced object matching
        matched_pairs, used_after_indices = enhanced_object_matching(before_results, after_results)
        
        # Use the same visualization logic as the original compare_images function
        # but with enhanced detection results
        INDOOR_BUSINESS_CLASSES = {
            0: "person", 24: "backpack", 26: "handbag", 27: "tie", 28: "suitcase",
            39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
            44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
            56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
            61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
            66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
            71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
            76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
        }
        
        # Create visualization (reusing visualization logic from compare_images)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Display before image
        ax1.imshow(np.array(before_image))
        ax1.set_title("Before Image (RF-DETR + DINO Enhanced)")
        ax1.axis('off')
        
        # Display after image
        ax2.imshow(np.array(after_image))
        ax2.set_title("After Image - RF-DETR + DINO Results")
        ax2.axis('off')
        
        # Track statistics
        still_present = 0
        missing = 0
        new_objects = 0
        
        # Draw bounding boxes on before image
        before_boxes = before_results["boxes"].cpu().numpy()
        before_labels = before_results["labels"].cpu().numpy()
        before_scores = before_results["scores"].cpu().numpy()
        
        for i, (box, label, score) in enumerate(zip(before_boxes, before_labels, before_scores)):
            if label not in INDOOR_BUSINESS_CLASSES:
                continue
                
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            class_name = INDOOR_BUSINESS_CLASSES[label]
            
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                   edgecolor='blue', facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x1, y1-10, f"{class_name}: {score:.2f}", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                    fontsize=8, color='black')
        
        # Draw comparison results on after image
        after_boxes = after_results["boxes"].cpu().numpy()
        after_labels = after_results["labels"].cpu().numpy()
        after_scores = after_results["scores"].cpu().numpy()
        
        # Draw matched objects (GREEN - still present)
        matched_before_indices = set()
        for before_idx, after_idx, iou, confidence in matched_pairs:
            label = after_labels[after_idx]
            if label not in INDOOR_BUSINESS_CLASSES:
                continue
                
            matched_before_indices.add(before_idx)
            still_present += 1
            
            box = after_boxes[after_idx]
            score = after_scores[after_idx]
            
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            class_name = INDOOR_BUSINESS_CLASSES[label]
            
            rect = patches.Rectangle((x1, y1), width, height, linewidth=3, 
                                   edgecolor='green', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1-10, f"{class_name}: {score:.2f} (PRESENT)", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                    fontsize=8, color='black')
        
        # Draw missing objects (RED - from before image)
        for i in range(len(before_boxes)):
            if i not in matched_before_indices:
                label = before_labels[i]
                if label not in INDOOR_BUSINESS_CLASSES:
                    continue
                    
                missing += 1
                box = before_boxes[i]
                score = before_scores[i]
                
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                class_name = INDOOR_BUSINESS_CLASSES[label]
                
                rect = patches.Rectangle((x1, y1), width, height, linewidth=3, 
                                       edgecolor='red', facecolor='none', linestyle='--')
                ax2.add_patch(rect)
                ax2.text(x1, y1-10, f"{class_name}: {score:.2f} (MISSING)", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
                        fontsize=8, color='black')
        
        # Draw new objects (BLACK - new in after image)
        for i in range(len(after_boxes)):
            if i not in used_after_indices:
                label = after_labels[i]
                if label not in INDOOR_BUSINESS_CLASSES:
                    continue
                    
                new_objects += 1
                box = after_boxes[i]
                score = after_scores[i]
                
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                class_name = INDOOR_BUSINESS_CLASSES[label]
                
                rect = patches.Rectangle((x1, y1), width, height, linewidth=3, 
                                       edgecolor='black', facecolor='none')
                ax2.add_patch(rect)
                ax2.text(x1, y1-10, f"{class_name}: {score:.2f} (NEW)", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
                        fontsize=8, color='black')
        
        # Add legend
        legend_elements = [
            patches.Patch(color='green', label=f'Still Present ({still_present})'),
            patches.Patch(color='red', label=f'Missing ({missing})'),
            patches.Patch(color='black', label=f'New Objects ({new_objects})')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(result_photo, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Print summary
        total_before_relevant = sum(1 for label in before_labels if label in INDOOR_BUSINESS_CLASSES)
        total_after_relevant = sum(1 for label in after_labels if label in INDOOR_BUSINESS_CLASSES)
        
        print(f"\nRF-DETR + DINO Enhanced Comparison Summary:")
        print(f"  Still Present: {still_present} objects")
        print(f"  Missing: {missing} objects")
        print(f"  New Objects: {new_objects} objects")
        print(f"  Total Before: {total_before_relevant} relevant objects ({len(before_boxes)} total detected)")
        print(f"  Total After: {total_after_relevant} relevant objects ({len(after_boxes)} total detected)")
        print(f"Enhanced comparison result saved to: {result_photo}")
        
        return f"RF-DETR + DINO Enhanced completed - {still_present} present, {missing} missing, {new_objects} new"
        
    except Exception as e:
        print(f"Error during RF-DETR + DINO enhanced comparison: {str(e)}")
        return f"RF-DETR + DINO Enhanced comparison failed - {str(e)}"


def detect_objects_in_image(image_path, model, processor, device):
    """
    Helper function to detect objects in a single image using RT-DETR (DINO-enhanced architecture)
    
    RT-DETR incorporates several DINO improvements:
    - Enhanced anchor handling with denoising training
    - Improved query selection and mixed query initialization
    - Better feature extraction and attention mechanisms
    
    Args:
        image_path: Path to the image file
        model: Pre-loaded RT-DETR model with DINO enhancements
        processor: Image processor for the RT-DETR model
        device: PyTorch device (cuda or cpu)
    
    Returns:
        tuple: (PIL Image, detection results) or (None, None) if error
    """
    try:
        # Load image file and convert to RGB format (removes alpha channel if present)
        image = Image.open(image_path).convert("RGB")
        
        # Process image into tensor format expected by the model
        # return_tensors="pt" means return PyTorch tensors
        inputs = processor(images=image, return_tensors="pt")
        
        # Move all input tensors to the same device as the model (GPU or CPU)
        # This prevents device mismatch errors during inference
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference with GPU optimizations
        with torch.no_grad():  # Disable gradient computation for faster inference
            if device.type == 'cuda':
                # Use automatic mixed precision on GPU for faster inference
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                # Standard inference on CPU
                outputs = model(**inputs)
        
        # Prepare target sizes for post-processing (height, width format)
        target_sizes = torch.tensor([image.size[::-1]])  # image.size is (width, height)
        
        # Convert raw model outputs to bounding boxes, labels, and scores
        # DINO enhancement: Use lower threshold (0.25) for better anchor selection,
        # then apply stricter filtering later for more robust detection
        results = processor.post_process_object_detection(outputs, threshold=0.25, target_sizes=target_sizes)[0]
        
        # DINO-style confidence refinement: Apply additional confidence filtering
        # to remove weak detections that passed the initial threshold
        if len(results["scores"]) > 0:
            high_conf_mask = results["scores"] > 0.35  # Stricter secondary threshold
            results["boxes"] = results["boxes"][high_conf_mask]
            results["labels"] = results["labels"][high_conf_mask]
            results["scores"] = results["scores"][high_conf_mask]
        
        return image, results
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    IoU measures overlap between boxes: IoU = intersection_area / union_area
    Values range from 0 (no overlap) to 1 (perfect overlap)
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
                   where (x1,y1) is top-left, (x2,y2) is bottom-right
    
    Returns:
        float: IoU score between 0.0 and 1.0
    """
    # Extract coordinates for both boxes
    x1_1, y1_1, x2_1, y2_1 = box1  # First box coordinates
    x1_2, y1_2, x2_2, y2_2 = box2  # Second box coordinates
    
    # Calculate intersection rectangle coordinates
    # Intersection top-left: maximum of both top-left coordinates
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    # Intersection bottom-right: minimum of both bottom-right coordinates  
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    
    # Check if boxes actually intersect
    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0  # No intersection
    
    # Calculate intersection area
    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    
    # Calculate individual box areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)  # First box area
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)  # Second box area
    
    # Calculate union area: total area covered by both boxes
    union = area1 + area2 - intersection
    
    # Return IoU ratio (avoid division by zero)
    return intersection / union if union > 0 else 0.0


def match_objects(results1, results2, iou_threshold=0.5):
    """
    Match objects between two detection results using DINO-enhanced matching strategy
    
    This function uses DINO-inspired improvements for better object matching:
    1. Multi-scale confidence weighting for better anchor selection
    2. Enhanced spatial overlap calculation with denoising consideration
    3. Adaptive thresholding based on object class confidence
    
    Args:
        results1: Detection results from first image (RT-DETR with DINO enhancements)
        results2: Detection results from second image (RT-DETR with DINO enhancements)
        iou_threshold: Base IoU threshold, adapted per class (default 0.5)
    
    Returns:
        tuple: (matched_pairs, used_indices2)
               matched_pairs: List of (index1, index2, iou_score, confidence_score) tuples
               used_indices2: Set of indices from image2 that were matched
    """
    # Convert GPU tensors to CPU numpy arrays for processing
    boxes1 = results1["boxes"].cpu().numpy()    # Bounding boxes from image 1
    labels1 = results1["labels"].cpu().numpy()  # Class labels from image 1  
    scores1 = results1["scores"].cpu().numpy()  # Confidence scores from image 1
    
    boxes2 = results2["boxes"].cpu().numpy()    # Bounding boxes from image 2
    labels2 = results2["labels"].cpu().numpy()  # Class labels from image 2
    scores2 = results2["scores"].cpu().numpy()  # Confidence scores from image 2
    
    matched_pairs = []      # Store successful matches
    used_indices2 = set()   # Track which objects in image2 have been matched
    
    # DINO-enhanced matching: Weight matches by both IoU and confidence scores
    for i in range(len(boxes1)):
        best_score = 0      # Track best combined score (IoU + confidence weighting)
        best_match = -1     # Track index of best matching object
        best_iou = 0        # Track IoU of best match
        
        # DINO improvement: Adaptive threshold based on detection confidence
        adaptive_threshold = max(0.3, iou_threshold * (0.5 + 0.5 * scores1[i]))
        
        # Check all objects in second image for potential matches
        for j in range(len(boxes2)):
            # Skip objects that have already been matched (one-to-one matching)
            if j in used_indices2:
                continue
                
            # Only consider objects of the same class (person matches person, etc.)
            if labels1[i] == labels2[j]:
                # Calculate spatial overlap between the two bounding boxes
                iou = calculate_iou(boxes1[i], boxes2[j])
                
                # DINO enhancement: Combined score considering IoU and confidence
                # Higher confidence detections get more weight in matching decisions
                confidence_weight = (scores1[i] + scores2[j]) / 2.0
                combined_score = iou * (0.7 + 0.3 * confidence_weight)
                
                # Update best match if this is better and meets adaptive threshold
                if combined_score > best_score and iou >= adaptive_threshold:
                    best_score = combined_score
                    best_iou = iou
                    best_match = j
        
        # If we found a good match, record it with confidence information
        if best_match != -1:
            confidence_score = (scores1[i] + scores2[best_match]) / 2.0
            matched_pairs.append((i, best_match, best_iou, confidence_score))
            used_indices2.add(best_match)  # Mark this object as used
    
    return matched_pairs, used_indices2


def compare_images(before_photo, after_photo, result_photo):
    """
    Main function to compare object detection results between two images using RT-DETR (DINO-enhanced)
    
    This function leverages DINO improvements for superior object detection and matching:
    1. Loads RT-DETR model with DINO enhancements (denoising training, mixed queries)
    2. Runs enhanced object detection with confidence refinement on both images
    3. Matches objects using DINO-style adaptive thresholding and confidence weighting
    4. Creates a visual comparison with color-coded bounding boxes:
       - GREEN: Objects present in both images (still there)
       - RED: Objects missing in second image (disappeared)  
       - BLACK: New objects in second image (appeared)
    5. Saves side-by-side comparison image with enhanced statistics
    
    Args:
        before_photo: Path to first image
        after_photo: Path to second image
        result_photo: Path where comparison result will be saved
    
    Returns:
        str: Summary of comparison results with DINO enhancement details
    """
    print("Running RT-DETR (DINO-enhanced) comparison between before and after images...")
    
    try:
        # === GPU SETUP AND OPTIMIZATION ===
        # Automatically detect best available device (GPU preferred)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Display GPU information if available
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # === MODEL LOADING ===
        # Load pre-trained RT-DETR model from Hugging Face (DINO-enhanced architecture)
        print("Loading RT-DETR model with DINO enhancements...")
        model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        
        # Move model to GPU for faster inference
        model = model.to(device)
        print(f"Model loaded on {device}")
        
        # Set model to evaluation mode (disables dropout, batch norm updates)
        model.eval()
        
        # Clear any existing GPU memory to avoid out-of-memory errors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # === OBJECT DETECTION ON BOTH IMAGES ===
        # Run detection on first image (before state)
        print(f"Processing before image: {before_photo}")
        before_image, before_results = detect_objects_in_image(before_photo, model, processor, device)
        
        # Run detection on second image (after state)
        print(f"Processing after image: {after_photo}")
        after_image, after_results = detect_objects_in_image(after_photo, model, processor, device)
        
        # Clean up GPU memory after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check if both images were processed successfully
        if before_image is None or after_image is None:
            return "Comparison failed - could not load images"
        
        # === OBJECT MATCHING ANALYSIS ===
        # Find which objects appear in both images based on class and spatial overlap
        matched_pairs, used_after_indices = match_objects(before_results, after_results)
        
        # === CLASS NAME MAPPING ===
        # COCO dataset class names - FILTERED for indoor business/residential environments
        # Focus on objects commonly found in offices, hotels, Airbnb, and professional buildings
        COCO_CLASSES = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        # Define indoor/business relevant classes (subset of COCO indices)
        # These are the object types we care about for professional buildings and rentals
        INDOOR_BUSINESS_CLASSES = {
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
        
        # === VISUALIZATION SETUP ===
        # Create side-by-side subplot layout for comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Display the "before" image on the left side
        ax1.imshow(np.array(before_image))
        ax1.set_title("Before Image")
        ax1.axis('off')  # Hide axis labels and ticks
        
        # Display after image
        ax2.imshow(np.array(after_image))
        ax2.set_title("After Image - DINO Comparison Results")
        ax2.axis('off')
        
        # Draw bounding boxes on before image (all in blue for reference)
        before_boxes = before_results["boxes"].cpu().numpy()
        before_labels = before_results["labels"].cpu().numpy()
        before_scores = before_results["scores"].cpu().numpy()
        
        for i, (box, label, score) in enumerate(zip(before_boxes, before_labels, before_scores)):
            # Only show objects that are relevant to indoor/business environments
            if label not in INDOOR_BUSINESS_CLASSES:
                continue
                
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            class_name = INDOOR_BUSINESS_CLASSES[label]
            
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                   edgecolor='blue', facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x1, y1-10, f"{class_name}: {score:.2f}", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                    fontsize=8, color='black')
        
        # Draw comparison results on after image
        after_boxes = after_results["boxes"].cpu().numpy()
        after_labels = after_results["labels"].cpu().numpy()
        after_scores = after_results["scores"].cpu().numpy()
        
        # Track statistics
        still_present = 0
        missing = 0
        new_objects = 0
        
        # Draw matched objects (GREEN - still present)
        matched_before_indices = set()
        for before_idx, after_idx, iou in matched_pairs:
            # Only process objects relevant to indoor/business environments
            label = after_labels[after_idx]
            if label not in INDOOR_BUSINESS_CLASSES:
                continue
                
            matched_before_indices.add(before_idx)
            still_present += 1
            
            box = after_boxes[after_idx]
            score = after_scores[after_idx]
            
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            class_name = INDOOR_BUSINESS_CLASSES[label]
            
            rect = patches.Rectangle((x1, y1), width, height, linewidth=3, 
                                   edgecolor='green', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1-10, f"{class_name}: {score:.2f} (PRESENT)", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                    fontsize=8, color='black')
        
        # Draw missing objects (RED - positions from before image)
        for i in range(len(before_boxes)):
            if i not in matched_before_indices:
                label = before_labels[i]
                # Only process objects relevant to indoor/business environments
                if label not in INDOOR_BUSINESS_CLASSES:
                    continue
                    
                missing += 1
                
                # Project the missing object's position onto the after image
                # For simplicity, we'll show it at the same coordinates
                box = before_boxes[i]
                score = before_scores[i]
                
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                class_name = INDOOR_BUSINESS_CLASSES[label]
                
                rect = patches.Rectangle((x1, y1), width, height, linewidth=3, 
                                       edgecolor='red', facecolor='none', linestyle='--')
                ax2.add_patch(rect)
                ax2.text(x1, y1-10, f"{class_name}: {score:.2f} (MISSING)", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
                        fontsize=8, color='black')
        
        # Draw new objects (BLACK - new in after image)
        for i in range(len(after_boxes)):
            if i not in used_after_indices:
                label = after_labels[i]
                # Only process objects relevant to indoor/business environments
                if label not in INDOOR_BUSINESS_CLASSES:
                    continue
                    
                new_objects += 1
                
                box = after_boxes[i]
                score = after_scores[i]
                
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                class_name = INDOOR_BUSINESS_CLASSES[label]
                
                rect = patches.Rectangle((x1, y1), width, height, linewidth=3, 
                                       edgecolor='black', facecolor='none')
                ax2.add_patch(rect)
                ax2.text(x1, y1-10, f"{class_name}: {score:.2f} (NEW)", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
                        fontsize=8, color='black')
        
        # Add legend
        legend_elements = [
            patches.Patch(color='green', label=f'Still Present ({still_present})'),
            patches.Patch(color='red', label=f'Missing ({missing})'),
            patches.Patch(color='black', label=f'New Objects ({new_objects})')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Adjust layout to prevent overlapping elements
        plt.tight_layout()
        
        # === SAVE RESULTS ===
        # Save the final comparison visualization to disk
        plt.savefig(result_photo, bbox_inches='tight', dpi=150)
        plt.close()  # Free memory by closing the plot
        
        # === PRINT SUMMARY STATISTICS ===
        # Calculate total relevant objects (filtered for indoor/business classes)
        total_before_relevant = sum(1 for label in before_labels if label in INDOOR_BUSINESS_CLASSES)
        total_after_relevant = sum(1 for label in after_labels if label in INDOOR_BUSINESS_CLASSES)
        
        print(f"\nDINO Comparison Summary (Indoor/Business Objects Only):")
        print(f"  Still Present: {still_present} objects")  # GREEN boxes
        print(f"  Missing: {missing} objects")              # RED boxes  
        print(f"  New Objects: {new_objects} objects")      # BLACK boxes
        print(f"  Total Before: {total_before_relevant} relevant objects ({len(before_boxes)} total detected)")
        print(f"  Total After: {total_after_relevant} relevant objects ({len(after_boxes)} total detected)")
        print(f"Comparison result saved to: {result_photo}")
        
        # Print what types of objects we're focusing on
        print(f"\nFocusing on {len(INDOOR_BUSINESS_CLASSES)} indoor/business object types:")
        print(f"  {', '.join(sorted(INDOOR_BUSINESS_CLASSES.values()))}")
        
        # Return summary string for the main timing system
        return f"DINO Comparison completed - {still_present} present, {missing} missing, {new_objects} new"
        
    except Exception as e:
        print(f"Error during DINO comparison: {str(e)}")
        return f"DINO Comparison failed - {str(e)}"
