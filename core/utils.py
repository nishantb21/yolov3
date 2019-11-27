import torch
import os

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