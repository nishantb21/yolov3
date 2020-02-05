
import torch
import os
import numpy as np
from loaders import COCODatasetLoader

def return_string(tag, loss, accuracy):
    string = "{0} Loss: {1:.3f} Accuracy: {2:.3f}"
    return string.format(tag, loss, accuracy)

def get_optimizer(parameters, optimizer_config):
    if optimizer_config.NAME == "Adam":
        return torch.optim.Adam(parameters, lr=optimizer_config.LEARNING_RATE)

    else:
        raise NotImplementedError("Optimizer not implemented")

def latest_run_folder(base_path):

    base_path = os.path.abspath(base_path)
    folder = os.listdir(base_path)
    run_number = len(folder) + 1
    run_folder_path = os.path.join(base_path, "run_{}".format(run_number))
    os.mkdir(run_folder_path)

    return run_folder_path

def create_run_artefacts(path):
    # Create weights folder
    weights_path = os.path.join(path, "weights")
    os.mkdir(weights_path)
    
    # Create log folder
    logs_path = os.path.join(path, "logs")
    os.mkdir(logs_path)

    # Create metadata.json
    metadata_path = os.path.join(path, "metadata.json")
    open(metadata_path, "w")

    return weights_path, logs_path, metadata_path
    
def create_and_get_paths(base_path):
    # Create the run folder
    new_path = latest_run_folder(base_path)

    # Create all the folders and files
    weights_path, logs_path, metadata_path = create_run_artefacts(new_path)

    return os.path.join(weights_path, "cp-{epoch:04d}.pt"), logs_path, metadata_path

def iou(bboxes_1, bboxes_2):
    # Get the left top position of the intersection area
    left_max = np.maximum(out_feature[..., :2], bboxes[..., :2])
    
    # Get the right bottom position of the intersection area
    right_min = np.minimum(out_feature[..., 2:4], bboxes[..., 2:4])
    
    # Calculate the deltas for area calulcation, reduce negative values to zero
    deltas = np.maximum(right_min - left_max, 0)
    
    # Calculate intersection and union
    intersection = deltas[..., 0] * deltas[..., 1]
    union = out_areas + bboxes_areas - intersection
    
    return intersection / union

def loss_od(outputs_target, output_predicted, anchors, factors, number_classes, lambda_coord=5, lambda_noobj=0.5):
    total_loss = 0.0
    for i in range(3):
        y = outputs_target[i]
        for j in range(3):
            length = (4 + 1 + number_classes)
            offset = length * j
            idx, idy = np.where(y[..., offset + 4] > 0)
            indexed_target = y[idx, idy, offset: offset + length]
            indexed_predicted = output_predicted[i][idx, idy, offset: offset + length]
            
            additive = np.squeeze(np.dstack((idx, idy)))
            indexed_target[..., :2] = (indexed_target[..., :2] + additive) * factors[i]
            indexed_predicted[..., :2] = (indexed_predicted[..., :2] + additive) * factors[i]

            indexed_target[..., 2:4] = np.exp(indexed_target[..., 2:4]) * anchors[(i * 3) + j]
            indexed_predicted[..., 2:4] = np.exp(indexed_predicted[..., 2:4]) * anchors[(i * 3) + j]

            indexed_target[..., :2] -= (indexed_target[..., 2:4] / 2)
            indexed_target[..., 2:4] += indexed_target[..., :2]
            indexed_predicted[..., :2] -= (indexed_predicted[..., 2:4] / 2)
            indexed_predicted[..., 2:4] += indexed_predicted[..., :2]

            squared_centers = np.power(indexed_predicted[..., :2] - indexed_target[..., :2], 2)
            center_loss = squared_centers[..., 0] + squared_centers[..., 1]

            squared_anchors = np.power(np.sqrt(indexed_predicted[..., 2:4]) - np.sqrt(indexed_target[..., 2:4]), 2)
            anchor_loss = squared_anchors[..., 0] + squared_anchors[..., 1]

            bbox_loss = (lambda_coord * np.sum(center_loss)) + (lambda_coord * np.sum(anchor_loss))

            objectness_loss = np.sum(np.power(indexed_predicted[..., 4] - indexed_target[..., 4], 2))

            classification_loss = np.sum(np.power((np.exp(indexed_predicted[..., 5:]) / np.sum(np.exp(indexed_predicted[..., 5:]))) - (np.exp(indexed_target[..., 5:]) / np.sum(np.exp(indexed_target[..., 5:]))), 2))

            idx, idy = np.where(y[..., offset + 4] == 0)
            indexed_target = y[idx, idy, offset + 4]
            indexed_predicted = output_predicted[i][idx, idy, offset + 4]
            non_objectness_loss = lambda_noobj * np.sum(np.power(indexed_predicted - indexed_target, 2))

            total_loss += (bbox_loss + objectness_loss + classification_loss + non_objectness_loss)


    return total_loss   

def mean_average_precision():
    pass

if __name__ == "__main__":
    root = "/data/nishantb/coco/val2017"
    annFile = "/data/nishantb/coco/annotations/instances_val2017.json"
    anchors = np.array([[ 20.1142666 ,  23.5920818 ],
       [ 61.54272606,  64.4243818 ],
       [ 78.43208441, 147.16067311],
       [173.79737463, 100.61049946],
       [130.99313932, 262.08766581],
       [419.42090078, 136.95277798],
       [267.46770012, 219.12108881],
       [258.48081984, 397.93091738],
       [461.50743771, 331.21772574]])

    factors = [8, 16, 32]
    obj = COCODatasetLoader(root, annFile, 512, 512, 80, anchors, factors)
    outputs = obj[0][1]
    outputs_predicted = np.copy(outputs)
    comp = loss_od(outputs, outputs, anchors, factors, 80)
    print(comp)