import os
import torch
from tqdm import tqdm
from torchvision import transforms, datasets

class ImageNetDatasetLoader():
    def __init__(self, root_dir, height, width, batch_size):
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        imagenet_dataset = datasets.ImageFolder(root=root_dir, transform=data_transform)
        self.dataset_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    def get_loader(self):
        return self.dataset_loader

if __name__ == "__main__":
    height = 256
    width = 256
    batch_size = 4
    root_dir = "./datasets/dummy"
    obj = ImageNetDatasetLoader(root_dir, height, width, batch_size)
    dataset = obj.get_loader()
    print(len(dataset))
    counter = 1
    tqdm_obj = tqdm(dataset, ncols=100)

    for x, y in tqdm_obj:
        tqdm_obj.set_description("Current iter {}".format(counter))
        counter += 1