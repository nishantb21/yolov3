import torch
from tqdm import tqdm
from utils import return_string

class Trainer():
    def __init__(self, model, optimizer, criterion, epochs, training_dataset_loader, validation_dataset_loader=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_dataset_loader = training_dataset_loader
        self.validation_dataset_loader = validation_dataset_loader
        self.epochs = epochs

    def train(self):
        for i in range(self.epochs):
            print("Epoch {} / {}".format(i + 1, self.epochs))
            self.model.train()
            n_total_count = 0
            n_correct_count = 0
            tqdm_obj = tqdm(self.training_dataset_loader, ncols=100)

            for x, y in tqdm_obj:
                n_total_count += x.shape[0]
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                predicted = self.model(x)
                loss = self.criterion(predicted, y)
                loss.backward()
                self.optimizer.step()
                n_correct_count += int(torch.sum(torch.argmax(predicted, 1) == y))
                tqdm_obj.set_description(desc=return_string("TRAINNING", loss, n_correct_count / n_total_count))

            if self.validation_dataset_loader:
                self.model.eval()
                n_total_count = 0
                n_correct_count = 0
                tqdm_obj = tqdm(self.validation_dataset_loader)
                with torch.no_grad():
                    for x, y in tqdm_obj:
                        n_total_count += x.shape[0]
                        x, y = x.to(self.device), y.to(self.device)
                        predicted = self.model(x)
                        loss = self.criterion(predicted, y)
                        n_correct_count += int(torch.sum(torch.argmax(predicted, 1) == y))
                        tqdm_obj.set_description(desc=return_string("TRAINNING", loss, n_correct_count / n_total_count))

class ImageNetBuilder():
    def __init__(self):
        pass

    def build(self):
        pass

if __name__ == "__main__":
    height = 256
    width = 256
    classes = 2
    batch_size = 2
    channels = 3
    root_dir = "./datasets/dummy"
    epochs = 1

    from models import DarkNet53
    from loaders import ImageNetDatasetLoader 

    net = DarkNet53(height, width, channels, classes)
    loaders_obj = ImageNetDatasetLoader(root_dir, height, width, batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    trainer_obj = Trainer(net, optimizer, criterion, epochs, loaders_obj.get_loader())
    trainer_obj.train()