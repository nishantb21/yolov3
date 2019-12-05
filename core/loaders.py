import os
import torch
from tqdm import tqdm
from torchvision import transforms, datasets

class DatasetLoader():
    def __init__(self):
        self.dataset_loader = None

    def get_loader(self):
        return self.dataset_loader

class ImageNetDatasetLoader(DatasetLoader):
    def __init__(self, root_dir, height, width, batch_size):
        super(ImageNetDatasetLoader, self).__init__()
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        imagenet_dataset = datasets.ImageFolder(root=root_dir, transform=data_transform)
        self.dataset_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

class COCODatasetLoader(DatasetLoader):
    def __init__(self):
        super(COCODatasetLoader, self).__init__()
        pass
    