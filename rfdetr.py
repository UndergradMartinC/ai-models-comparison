# Import required libraries
import time                                                    # For timing operations
import torch                                                   # PyTorch for deep learning
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor  # Hugging Face RF-DETR model
from PIL import Image                                          # Image loading and processing
import cv2                                                     # OpenCV for computer vision
import numpy as np                                             # Numerical operations
import matplotlib.pyplot as plt                               # Plotting and visualization
import matplotlib.patches as patches                          # Drawing shapes on plots

def rfdetr(before_photo="before.jpg", after_photo="after.jpg", result_photo="rfdetr_result.jpg"):
    """
    Main RF-DETR function for object detection comparison
    This is the entry point called by main.py for timing comparisons
    """
    print("Running RF-DETR model...")
    
    # Call the main comparison function with the image paths
    result = compare_images(before_photo, after_photo, result_photo)
    
    # Return result string for the timing system in main.py
    return result


def detect_objects_in_image(image_path, model, processor, device):
    """
    Helper function to detect objects in a single image
    
    Args:
        image_path: Path to the image file
        model: Pre-loaded RF-DETR model
        processor: Image processor for the model
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
        # threshold=0.3 means only keep detections with >30% confidence
        results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
        
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
    Match objects between two detection results based on IoU and class similarity
    
    This function determines which objects in image 1 correspond to objects in image 2
    by finding the best matches based on:
    1. Same object class (e.g., person, car, etc.)
    2. Spatial overlap (IoU) above threshold
    
    Args:
        results1: Detection results from first image
        results2: Detection results from second image  
        iou_threshold: Minimum IoU required to consider objects as matches (default 0.5)
    
    Returns:
        tuple: (matched_pairs, used_indices2)
               matched_pairs: List of (index1, index2, iou_score) tuples
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
    
    # For each object in first image, find the best match in second image
    for i in range(len(boxes1)):
        best_iou = 0        # Track best overlap score found so far
        best_match = -1     # Track index of best matching object
        
        # Check all objects in second image for potential matches
        for j in range(len(boxes2)):
            # Skip objects that have already been matched (one-to-one matching)
            if j in used_indices2:
                continue
                
            # Only consider objects of the same class (person matches person, etc.)
            if labels1[i] == labels2[j]:
                # Calculate spatial overlap between the two bounding boxes
                iou = calculate_iou(boxes1[i], boxes2[j])
                
                # Update best match if this is better and meets minimum threshold
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = j
        
        # If we found a good match, record it
        if best_match != -1:
            matched_pairs.append((i, best_match, best_iou))
            used_indices2.add(best_match)  # Mark this object as used
    
    return matched_pairs, used_indices2


def compare_images(before_photo, after_photo, result_photo):
    """
    Main function to compare object detection results between two images
    
    This function:
    1. Loads the RF-DETR model with GPU optimization
    2. Runs object detection on both images
    3. Matches objects between images using IoU and class similarity
    4. Creates a visual comparison with color-coded bounding boxes:
       - GREEN: Objects present in both images (still there)
       - RED: Objects missing in second image (disappeared)  
       - BLACK: New objects in second image (appeared)
    5. Saves side-by-side comparison image with statistics
    
    Args:
        before_photo: Path to first image
        after_photo: Path to second image
        result_photo: Path where comparison result will be saved
    
    Returns:
        str: Summary of comparison results
    """
    print("Running RF-DETR comparison between before and after images...")
    
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
        # Load pre-trained RF-DETR model from Hugging Face
        print("Loading RF-DETR model...")
        model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
        processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        
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
        ax2.set_title("After Image - Comparison Results")
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
        
        print(f"\nComparison Summary (Indoor/Business Objects Only):")
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
        return f"Comparison completed - {still_present} present, {missing} missing, {new_objects} new"
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        return f"Comparison failed - {str(e)}"