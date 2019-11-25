class Trainer():
    def __init__(self, model, optimizer, criterion, training_dataset_loader, validation_dataset_loader, epochs):
        self.model = model
        self.optimizer = optimizer
        self.training_dataset_loader = training_dataset_loader
        self.validation_dataset_loader = validation_dataset_loader
        self.epochs = epochs

    def train(self):
        for i in range(epochs):
            self.model.train()
            for batch_idx, (x, y) in enumerate(self.training_dataset_loader):
                self.optimizer.zero_grad()
                predicted = self.model(x)
                loss = self.criterion(predicted, y)
                loss.backward()
                self.optimizer.step()

                pass

            for batch_idx, (x, y) in enumerate(self.validation_dataset_loader):
                pass

class ImageNetBuilder():
    def __init__(self):
        pass

    def build(self):
        pass

