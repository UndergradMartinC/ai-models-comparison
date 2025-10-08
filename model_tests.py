import json
from COCO_CLASSES import INDOOR_BUSINESS_CLASSES
import numpy as np
import sympy

'''
This script contains the functions to test the each model against the ground truth data. The class ConfusionMatrix 
is used to make the confusion matrix for each class. This is used to calculate the precision, accuracy, and sensitivity 
for each class.
'''


IOU_THRESHOLD = 0.5
AREA_THRESHOLD = 0.5

def area_is_similar(bbox1, bbox2):
    """
    Calculate the area difference ratio between two bounding boxes
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    areaA = (x2 - x1) * (y2 - y1)
    areaB = (x4 - x3) * (y4 - y3)

    if areaA > areaB:
        return areaA - areaB / areaA < AREA_THRESHOLD
    else:
        return areaB - areaA / areaB < AREA_THRESHOLD


def iou_is_within_threshold(bbox1, bbox2):
    """
    Calculate Intersection Over Union (IoU) between two bounding boxes
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # Intersection
    ix1 = max(x1, x3)
    iy1 = max(y1, y3)
    ix2 = min(x2, x4)
    iy2 = min(y2, y4)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    # Individual areas
    areaA = (x2 - x1) * (y2 - y1)
    areaB = (x4 - x3) * (y4 - y3)

    # Union = sum - intersection
    union_area = areaA + areaB - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0

    return iou < IOU_THRESHOLD





def is_object_present(object, reference_json):
    """
    Compare ground truth and model predictions
    """
    for item in reference_json:
        if item['class'] == object.class_name and iou(object.bbox, item.bbox) > IOU_THRESHOLD:
            return True
    return False

class Object:
    def __init__(self, class_name):
        self.class_name = class_name
        self.precision = 0
        self.sensitivity = 0
        self.specificity = 0
        self.f1_score = 0
        self.average_precision = 0

    def set_precision(self, precision):
        self.precision = precision
    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity
    def set_specificity(self, specificity):
        self.specificity = specificity
    def set_f1_score(self, f1_score):
        self.f1_score = f1_score
    def set_average_precision(self, average_precision):
        self.average_precision = average_precision


class ConfusionMatrix:
    def __init__(self, class_name_array, reference_json):
        self.reference_json = reference_json #need logic for reference json to be an array of objects
        self.confusion_matrix = self.make_confusion_matrix()
        self.object_array = self.make_object_array()
        self.mean_average_precision = 0
        self.mean_f1_score = 0

    def make_object_array(self):
        object_array = []
        for item in self.reference_json:
            object_array.append(Object(item['class']))
        return object_array
    
    def make_confusion_matrix(self):
        confusion_matrix = np.zeros((reference_json.length, reference_json.length))
        return confusion_matrix

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_class_index(self, class_name):
        return self.class_name_array.index(class_name)

    def get_accuracy(self, true_positives, false_positives, false_negatives, true_negatives):
        return (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)

    def get_precision(self, true_positives, false_positives):
        return true_positives / (true_positives + false_positives)

    def get_sensitivity(self, true_positives, false_negatives):
        return true_positives / (true_positives + false_negatives)

    def get_specificity(self, true_negatives, false_positives):
        return true_negatives / (true_negatives + false_positives)

    def get_f1_score(self, precision, sensitivity):
        return 2 * precision * sensitivity / (precision + sensitivity)

    def get_average_precision(self, precision):
        #intgrate precision function
        return sympy.integrate(precision, 0, 1)

    def get_mean_average_precision(self, json_array):
        mean_average_precision = 0
        for item in json_array:
            mean_average_precision += self.get_average_precision(item['precision'])
        return mean_average_precision / len(json_array)

    def set_class_metrics(self, class_name):
        int true_positives = 0
        int false_positives = 0
        int false_negatives = 0
        int true_negatives = 0

        int class_index = self.get_class_index(class_name)

        for i in range(self.confusion_matrix.length):
            for j in range(self.confusion_matrix.length):
                if i == class_index and j == class_index:
                    true_positives += self.confusion_matrix[i][j]
                elif i == class_index:
                    false_positives += self.confusion_matrix[i][j]
                elif j == class_index:
                    false_negatives += self.confusion_matrix[i][j]
                else:
                    true_negatives += self.confusion_matrix[i][j]

        self.object_array[class_index].set_precision(self.get_precision(true_positives, false_positives))
        self.object_array[class_index].set_sensitivity(self.get_sensitivity(true_positives, false_negatives))
        self.object_array[class_index].set_specificity(self.get_specificity(true_negatives, false_positives)) 
        self.object_array[class_index].set_f1_score(self.get_f1_score(self.object_array[class_index].precision, self.object_array[class_index].sensitivity))
        self.object_array[class_index].set_average_precision(self.get_average_precision(self.object_array[class_index].precision))

        return self.object_array[class_index].precision, self.object_array[class_index].sensitivity, self.object_array[class_index].specificity, self.object_array[class_index].f1_score, self.object_array[class_index].average_precision

    def increment_cell(self, for_class, to_class):
        self.confusion_matrix[self.get_class_index(for_class)][self.get_class_index(to_class)] += 1

    def get_mean_average_precision(self):
        mean_average_precision = 0
        for item in self.object_array:
            mean_average_precision += item.precision
            
        self.mean_average_precision = mean_average_precision / len(self.object_array)
        return mean_average_precision / len(self.object_array)

    def get_mean_f1_score(self):
        mean_f1_score = 0
        for item in self.object_array:
            mean_f1_score += item.f1_score
        
        self.mean_f1_score = mean_f1_score / len(self.object_array)
        return mean_f1_score / len(self.object_array)

'''
def model_test(image_name="sample_test.jpg", class_array=["chair", "table", "sofa"]):
    """
    Compare AI model predictions against ground truth data
    
    Args:
        image_name: Name of the image being tested (e.g., "sample.jpg")
        class_array: List of objects detected by the AI model (e.g., ["person", "car", "person"])
    
    Returns:
        tuple: (precision, accuracy, sensitivity)
    """
    # JSON file contains GROUND TRUTH (what we know is actually in the image)
    # class_array contains MODEL PREDICTIONS (what the AI detected)
    
    #precision = true positives / (true positives + false positives)
    #accuracy = true positives / (true positives + false positives + false negatives)  
    #sensitivity = true positives / (true positives + false negatives)

    # Load ground truth data from JSON file

    #convert image name to json name
    json_name = image_name.replace(".jpg", ".json")

    try:
        with open(json_name, 'r') as file:
            ground_truth_data = json.load(file)
    except FileNotFoundError:
        print("File not found")
        return 0, 0, 0
    
    # Convert ground truth to dictionary {class: instances}
    ground_truth = {item['class']: item['instances'] for item in ground_truth_data}
    
    # Convert model predictions (class_array) to dictionary {class: instances}
    model_predictions = {}
    for class_name in class_array:
        model_predictions[class_name] = model_predictions.get(class_name, 0) + 1
    
    # Get COCO class names for true negative calculation
    coco_class_names = set(INDOOR_BUSINESS_CLASSES.values())
    
    # Get all unique classes from ground truth, model predictions, and COCO classes
    all_classes = coco_class_names
    
    # Calculate metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    for class_name in all_classes:
        gt_count = ground_truth.get(class_name, 0)
        pred_count = model_predictions.get(class_name, 0)
        
        # For each class, calculate TP, FP, FN, TN based on instance counts
        if gt_count > 0 and pred_count > 0:
            # Both have instances - TP is minimum of both
            true_positives += min(gt_count, pred_count)
            # FP is excess predictions
            if pred_count > gt_count:
                false_positives += pred_count - gt_count
            # FN is missed ground truth instances
            if gt_count > pred_count:
                false_negatives += gt_count - pred_count
        elif gt_count > 0 and pred_count == 0:
            # Ground truth has instances but no predictions - all FN
            false_negatives += gt_count
        elif gt_count == 0 and pred_count > 0:
            # Predictions but no ground truth - all FP
            false_positives += pred_count
        elif gt_count == 0 and pred_count == 0 and class_name in coco_class_names:
            # COCO class not in ground truth and not predicted - True Negative
            true_negatives += 1
    
    # Calculate metrics with division by zero protection
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    
    if true_positives + false_negatives > 0:
        sensitivity = true_positives / (true_positives + false_negatives)
    else:
        sensitivity = 0
    
    # Calculate accuracy using all four metrics including true negatives from COCO classes
    total_all = true_positives + false_positives + false_negatives + true_negatives
    if total_all > 0:
        accuracy = (true_positives + true_negatives) / total_all
    else:
        accuracy = 0
    
    return precision, accuracy, sensitivity


'''