import json
from COCO_CLASSES import INDOOR_BUSINESS_CLASSES
import numpy as np

'''
This script contains the functions to test the each model against the ground truth data. The class ConfusionMatrix 
is used to make the confusion matrix for each class. This is used to calculate the precision, accuracy, and sensitivity 
for each class.
'''


IOU_THRESHOLD = 0.5
AREA_THRESHOLD = 0.5

class Object:

    def __init__(self, class_name, bbox):
        self.class_name = class_name
        self.bbox = bbox

    def get_area(self):
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

class ObjectMetrics:
    def __init__(self, class_name):
        self.class_name = class_name
        self.precision = 0
        self.sensitivity = 0
        self.specificity = 0
        self.f1_score = 0

    def set_precision(self, precision):
        self.precision = precision
    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity
    def set_specificity(self, specificity):
        self.specificity = specificity
    def set_f1_score(self, f1_score):
        self.f1_score = f1_score


class ConfusionMatrix:
    def __init__(self, reference_json):
        
        self.index_dict = {}
        for i in range(len(INDOOR_BUSINESS_CLASSES)):
            self.index_dict[INDOOR_BUSINESS_CLASSES[i]] = i
        
        
        self.num_classes = len(INDOOR_BUSINESS_CLASSES)


        self.reference_json = reference_json #need logic for reference json to be an array of objects)
        self.confusion_matrix = self.make_confusion_matrix()
        self.reference_object_array = self.make_object_array()
        self.class_metrics_array = self.make_class_metrics_array()
        self.mean_average_precision = 0
        self.mean_f1_score = 0
        self.unmatched_objects = []
        self.missing_objects = []

    def make_class_metrics_array(self):
        class_metrics_array = []
        for class_name in self.class_name_array:
            class_metrics_array.append(ObjectMetrics(class_name))
        return class_metrics_array

    def make_object_array(self):
        object_array = []
        for item in self.reference_json:
            object_array.append(Object(item['class'], item['bbox']))
        return object_array
    
    def make_confusion_matrix(self):
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        return confusion_matrix

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_class_index(self, class_name):
        return self.index_dict[class_name]

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


    def get_matrix_metrics(self):
        for item in self.reference_object_array:
            class_name = item.class_name
            self.set_class_metrics(class_name)
        
        self.set_mean_average_precision()
        self.set_mean_f1_score()

        return self.class_metrics_array, self.mean_average_precision, self.mean_f1_score




        

    def get_mean_average_precision(self):
        mean_average_precision = 0
        for item in self.object_array:
            mean_average_precision += item.precision
        return mean_average_precision / len(self.object_array)

    def set_class_metrics(self, class_name):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        class_index = self.get_class_index(class_name)

        for i in range(len(self.confusion_matrix)):
            for j in range(len(self.confusion_matrix)):
                if i == class_index and j == class_index:
                    true_positives += self.confusion_matrix[i][j]
                elif i == class_index:
                    false_positives += self.confusion_matrix[i][j]
                elif j == class_index:
                    false_negatives += self.confusion_matrix[i][j]
                else:
                    true_negatives += self.confusion_matrix[i][j]
            
        for object in self.unmatched_objects:
            if object.class_name == class_name:
                false_positives += 1
            else:
                true_negatives += 1

        self.object_array[class_index].set_precision(self.get_precision(true_positives, false_positives))
        self.object_array[class_index].set_sensitivity(self.get_sensitivity(true_positives, false_negatives))
        self.object_array[class_index].set_specificity(self.get_specificity(true_negatives, false_positives)) 
        self.object_array[class_index].set_f1_score(self.get_f1_score(self.object_array[class_index].precision, self.object_array[class_index].sensitivity))

        return self.object_array[class_index].precision, self.object_array[class_index].sensitivity, self.object_array[class_index].specificity, self.object_array[class_index].f1_score

    def increment_cell(self, reference_class, object_class):
        self.confusion_matrix[self.get_class_index(reference_class)][self.get_class_index(object_class)] += 1

    def set_mean_average_precision(self):
        mean_average_precision = 0
        for item in self.object_array:
            mean_average_precision += item.precision
            
        self.mean_average_precision = mean_average_precision / len(self.object_array)
        return mean_average_precision / len(self.object_array)

    def set_mean_f1_score(self):
        mean_f1_score = 0
        for item in self.object_array:
            mean_f1_score += item.f1_score
        
        self.mean_f1_score = mean_f1_score / len(self.object_array)
        return mean_f1_score / len(self.object_array)

    def area_is_similar(self, object, reference_object):
        """
        Calculate the area difference ratio between two bounding boxes
        """

        area1 = object.get_area()
        area2 = reference_object.get_area()

        if area1 > area2:
            return (area1 - area2) / area1 < AREA_THRESHOLD
        else:
            return (area2 - area1) / area2 < AREA_THRESHOLD

    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection Over Union (IoU) between two bounding boxes
        Returns the actual IoU value
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
        return iou


    def is_object_present(self, object):
        """
        Compare ground truth and model predictions
        """
        for reference_object in self.reference_object_array:
            if (self.calculate_iou(object.bbox, reference_object.bbox) > IOU_THRESHOLD and self.area_is_similar(object, reference_object)):
                self.increment_cell(reference_object.class_name, object.class_name)
                return True
        return False

    def handle_object_data(self, class_name, bbox, predicted_class_name):
        detected_object = Object(class_name, bbox)
        if self.is_object_present(detected_object, self.reference_object_array):
            self.increment_cell(detected_object.class_name, predicted_class_name)
        else:
            self.unmatched_objects.append(detected_object)

