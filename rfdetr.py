import time
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def rfdetr():
    """Main RF-DETR function for object detection"""
    print("Running RF-DETR model...")
    
    try:
        # Load the model and processor
        print("Loading RF-DETR model...")
        model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
        processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        
        # Load and preprocess the image
        image_path = "sample.jpg"
        print(f"Loading image: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: {image_path} not found. Please ensure the image exists in the current directory.")
            return "RF-DETR failed - image not found"
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Run inference
        print("Running inference...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process the outputs
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
        
        # Display results
        print(f"Detected {len(results['scores'])} objects:")
        
        # Convert image to numpy array for visualization
        img_array = np.array(image)
        
        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_array)
        
        # COCO class names (RF-DETR is typically trained on COCO)
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
        
        for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            box = box.cpu().numpy()
            score = score.cpu().numpy()
            label = label.cpu().numpy()
            
            # Get class name
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
            
            print(f"  {i+1}. {class_name}: {score:.3f} confidence")
            print(f"     Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle patch
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                   edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1-10, f"{class_name}: {score:.2f}", 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=10, color='black')
        
        ax.set_title(f"RF-DETR Object Detection - {len(results['scores'])} objects detected")
        ax.axis('off')
        
        # Save the result
        output_path = "rfdetr_detection_result.jpg"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Detection result saved to: {output_path}")
        
        return f"RF-DETR completed - detected {len(results['scores'])} objects"
        
    except Exception as e:
        print(f"Error during RF-DETR inference: {str(e)}")
        return f"RF-DETR failed - {str(e)}"