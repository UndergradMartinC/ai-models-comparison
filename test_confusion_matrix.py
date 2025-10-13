#!/usr/bin/env python3
"""
Test script for confusion matrix functionality without ML dependencies
"""

import json
from COCO_CLASSES import INDOOR_BUSINESS_CLASSES, CONFIDENCE_THRESHOLD
from model_tests import ConfusionMatrix


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
    
    # Check what classes are in the index_dict
    print(f"\nDEBUG - Available classes in confusion matrix:")
    print(f"  Classes in index_dict: {list(confusion_matrix.index_dict.keys())}")
    print(f"  Total classes: {len(confusion_matrix.index_dict)}")
    
    # Process each detected object through confusion matrix
    print(f"\nProcessing detections through confusion matrix...")
    for obj in simulated_detections:
        print(f"\n  Processing: {obj['class']} at {obj['bbox']}")
        confusion_matrix.handle_object_data(obj['class'], obj['bbox'])
    
    # Check final state
    print(f"\nFINAL STATE:")
    print(f"  Confusion matrix sum: {confusion_matrix.get_confusion_matrix().sum()}")
    print(f"  Unmatched objects: {len(confusion_matrix.unmatched_objects)}")
    
    # Print the actual confusion matrix
    cm = confusion_matrix.get_confusion_matrix()
    print(f"\nCONFUSION MATRIX ({cm.shape[0]}x{cm.shape[1]}):")
    print("=" * 80)
    
    # Print header with class names (first 10 classes for readability)
    classes = list(confusion_matrix.index_dict.keys())
    max_classes_to_show = min(10, len(classes))
    
    print(f"{'':>15}", end="")
    for j in range(max_classes_to_show):
        print(f"{classes[j][:8]:>8}", end="")
    print()
    
    # Print matrix rows
    for i in range(max_classes_to_show):
        print(f"{classes[i][:15]:>15}", end="")
        for j in range(max_classes_to_show):
            print(f"{cm[i,j]:>8.0f}", end="")
        print()
    
    if len(classes) > max_classes_to_show:
        print(f"... (showing first {max_classes_to_show} of {len(classes)} classes)")
    
    print("=" * 80)
    
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
    
    # Show detailed class metrics
    print(f"\nClass-Specific Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1 Score':<10}")
    print("-" * 60)
    
    for metric in class_metrics:
        print(f"{metric.class_name:<15} {metric.precision:<10.4f} {metric.sensitivity:<12.4f} "
              f"{metric.specificity:<12.4f} {metric.f1_score:<10.4f}")
    
    print("="*60)
    
    return mean_ap, mean_f1


if __name__ == "__main__":
    # Test confusion matrix with dummy data
    mean_ap, mean_f1 = test_confusion_matrix()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Mean Average Precision: {mean_ap:.4f}")
    print(f"   Mean F1 Score: {mean_f1:.4f}")
