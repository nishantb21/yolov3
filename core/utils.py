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

def resize_images(image, bboxes, target_width, target_height):
    from PIL import ImageOps
    from matplotlib import pyplot as plt
    from matplotlib import patches

    original_image = image
    fig, ax = plt.subplots(1)
    ax.imshow(original_image)
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r')
        ax.add_patch(rect)
    plt.show()

    width, height = image.size
    if width > height:
        scale = target_width / width
        new_width = target_width
        new_height = int(scale * height )
        padding = (target_height - new_height) // 2
        image = image.resize((new_width, new_height))
        image = ImageOps.expand(image, border=(0, padding, 0, padding), fill=0)
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for bbox in bboxes:
            bbox = [scale * i for i in bbox]
            rect = patches.Rectangle((bbox[0], bbox[1] + padding), bbox[2], bbox[3], linewidth=1, edgecolor='r')
            ax.add_patch(rect)
        plt.show()

    elif height > width:
        scale = target_height / height


    


if __name__ == "__main__":
    from PIL import Image
    height = 512
    width = 512
    img = Image.open("sample.jpg")
    bboxes = [[236.98,142.51,24.7,69.5], [7.03,167.76,149.32,94.87], [557.21,209.19,81.35,78.73]]
    resize_images(img, bboxes, width, height)