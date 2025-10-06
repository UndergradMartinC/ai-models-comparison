import json

def model_test(image_name, class_array):
    #example input: image_name="sample.jpg", class_array=["chair", "table", "sofa"]
    # load json file like sameple_test.json
    # convert array to a dictionary with key as class and value as instances
    # compare the dictionary with the json file
    # return the precision, accuracy, sensitivity
    precision = 0
    accuracy = 0
    sensitivity = 0
    return precision, accuracy, sensitivity
