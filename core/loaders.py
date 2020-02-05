import os
import torch
from tqdm import tqdm
from torchvision import transforms, datasets
from PIL import ImageOps
import numpy as np 

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

class COCODataset(datasets.CocoDetection):
    def __init__(self, root, annFile, height, width, number_classes, anchors, scaling_factors, transform):
        super(COCODatasetLoader, self).__init__(root, annFile)
        self.height = height
        self.width = width 
        self.number_classes = number_classes
        self.anchors = anchors
        self.scaling_factors = scaling_factors
        self.number_classes = number_classes
        self.transform = transform

    def __len__(self):
        return super(COCODatasetLoader, self).__len__()

    def resize_image(self, image, bboxes):
        curr_width, curr_height = image.size
        bboxes = np.array(bboxes)
        
        if curr_width > curr_height:
            scale = self.width / curr_width
            new_width = self.width
            new_height = int(scale * curr_height)
            padding = (self.height - new_height) // 2
            padding_tuple = (0, padding, 0, padding)

        else:
            scale = self.height / curr_height
            new_height = self.height
            new_width = int(scale * curr_width)
            padding = (self.width - new_width) // 2
            padding_tuple = (padding, 0, padding, 0)

        image = image.resize((new_width, new_height))
        image = ImageOps.expand(image, border=padding_tuple, fill=0)

        padding_tuple = np.array(padding_tuple[:2])
        bboxes = bboxes * scale
        bboxes[..., :2] += padding_tuple

        return image, bboxes

    def get_closest_anchor(self, bbox):
        def map_fn(anchor):
            return np.absolute(anchor[0] - bbox[2]) + np.absolute(anchor[1] - bbox[3])

        output = np.apply_along_axis(map_fn, 1, self.anchors)
        return np.argmin(output)

    def create_output_matrices(self, bboxes, classes):
        output_sizes = [np.array([self.width, self.height]) // i for i in self.scaling_factors]
        outputs = [np.zeros(np.append(size, (4 + 1 + self.number_classes) * 3)) for size in output_sizes]
        print(bboxes.shape[0])
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            anchor_idx = self.get_closest_anchor(bboxes[i])
            factor_idx = anchor_idx // 3
            factor = self.scaling_factors[factor_idx]
            bbox[:2] = bbox[:2] / (bbox[2:] / 2)
            softmax_classes = np.zeros(self.number_classes)
            softmax_classes[classes[i] - 1] = 1.0
            scaled = bbox[:2] / factor
            indices = np.array(np.floor_divide(bbox[:2], factor), dtype="int")
            offset = (anchor_idx % 3) * (4 + 1 + self.number_classes)
            print("Writing to ", indices[0], indices[1], offset, factor_idx)
            outputs[factor_idx][indices[0], indices[1], offset:(offset + 2)] = scaled - indices
            outputs[factor_idx][indices[0], indices[1], (offset + 2):(offset + 4)] = np.log(bbox[2:4] / self.anchors[anchor_idx])
            outputs[factor_idx][indices[0], indices[1], (offset + 4)] = 1.0
            outputs[factor_idx][indices[0], indices[1], (offset + 5):(offset + 5 + self.number_classes)] = softmax_classes

        return outputs

    def __getitem__(self, idx):
        image, metadata = super(COCODatasetLoader, self).__getitem__(idx)
        metadata = [(i["bbox"], i["category_id"]) for i in metadata if i["category_id"] <= self.number_classes]
        bboxes, classes = zip(*metadata)
        image, bboxes = self.resize_image(image, bboxes)
        outputs = self.create_output_matrices(bboxes, classes)

        return self.transform(image), transforms.ToTensor()(outputs)

class COCODatasetLoader(DatasetLoader):
    def __init__(self, root, annFile, height, width, number_classes, anchors, scaling_factors):
        super(DatasetLoader).__init__()
        data_transform


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
    for y in outputs:
        for j in range(3):
            offset = (4 + 1 + 80) * j
            idx, idy = np.where(y[..., offset + 4] > 0)
            print(idx.shape, idy.shape)
            print(np.squeeze(np.dstack((idx, idy))).shape)
            output = y[idx, idy, offset: offset + 4]
            print(output.shape)