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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Check for Grounding DINO availability
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    GROUNDINGDINO_AVAILABLE = True
    print("✅ Grounding DINO available!")
except ImportError:
    GROUNDINGDINO_AVAILABLE = False
    print("⚠️  Grounding DINO not available")


class GroundingDINODetector:
    """Grounding DINO for open vocabulary detection"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.box_threshold = 0.35
        self.text_threshold = 0.35
        self.confidence_threshold = 0.35
        
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
            print("⚠️  Grounding DINO not available")
            return False
        
        try:
            weights_path = "weights/groundingdino_swint_ogc.pth"
            config_path = "weights/GroundingDINO_SwinT_OGC.py"
            
            if not all(os.path.exists(p) for p in [weights_path, config_path]):
                print("❌ Grounding DINO model files not found")
                return False
            
            self.model = load_model(config_path, weights_path, device="cpu")
            print("✅ Grounding DINO loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Grounding DINO: {e}")
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
            print("❌ Grounding DINO model not loaded")
            return []
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        try:
            print(f"🔍 Running Grounding DINO on {image_path}")
            
            # Load image
            image_source, image = load_image(image_path)
            h, w, _ = image_source.shape
            
            # Use custom prompts or comprehensive list
            prompts = custom_prompts if custom_prompts else self.get_comprehensive_prompts()
            
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
                    
                    # Validate
                    if x2 > x1 and y2 > y1:
                        detection = {
                            "object": label.strip().lower(),
                            "confidence": float(confidence),
                            "bbox": [x1, y1, x2, y2],
                            "detection_type": "grounding_dino"
                        }
                        detections.append(detection)
            
            print(f"✅ Grounding DINO detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            print(f"❌ Grounding DINO detection failed: {e}")
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
            label = f"🎯 {obj}: {conf:.2f}"
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
            print(f"💾 Visualization saved: {save_path}")
        
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
        print(f"💾 Grounding DINO results saved: {json_path}")
        
        # Save visualization
        vis_path = f"{output_dir}/dino_{base_name}_visualization.jpg"
        self.create_visualization(image_path, detections, vis_path)
        
        return json_path, vis_path


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
    print("🎯 GROUNDING DINO COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    detector = GroundingDINODetector()
    
    if detector.model is None:
        print("❌ Cannot run - Grounding DINO not available")
        return
    
    # Check if images exist
    for img_path in [image1_path, image2_path]:
        if not os.path.exists(img_path):
            print(f"❌ {img_path} not found")
            return
    
    results = {}
    
    # Analyze both images
    for img_path in [image1_path, image2_path]:
        print(f"\n📸 Analyzing {img_path}")
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
        
        print(f"✅ Objects detected: {len(detections)}")
        print(f"📁 Results saved: {json_path}")
        print(f"🖼️  Visualization: {vis_path}")
        print(f"📊 Categories: {category_counts}")
        
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
        print("🔍 COMPARISON")
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
    print("📈 GROUNDING DINO SUMMARY")  
    print(f"{'='*60}")
    print(f"📁 Results saved to: outputs/dino_*")
    print(f"🎯 Analysis complete!")


if __name__ == "__main__":
    # Run comprehensive DINO analysis
    comprehensive_dino_analysis()
