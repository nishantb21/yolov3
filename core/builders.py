import torch
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm
from core.utils import return_string, create_and_get_paths, get_optimizer
from core.models import backbones
from core.loaders import ImageNetDatasetLoader

class Trainer():
    def __init__(self, model, optimizer, criterion, epochs, training_dataset_loader, validation_dataset_loader=None, lr_scheduler=None, weights_path=None, logs_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_dataset_loader = training_dataset_loader
        self.validation_dataset_loader = validation_dataset_loader
        self.epochs = epochs
        self.weights_path = weights_path
        self.logs_path = logs_path
        self.lr_scheduler = lr_scheduler

    def train(self):
        if self.logs_path:
            summary_writer = SummaryWriter(log_dir=self.logs_path)

        for i in range(self.epochs):
            iteration = i + 1
            print("Epoch {} / {}".format(iteration, self.epochs))
            self.model.train()
            n_total_count = 0
            n_correct_count = 0
            tqdm_obj = tqdm(self.training_dataset_loader, ncols=100)
            counter = len(tqdm_obj) * i

            for x, y in tqdm_obj:
                counter += 1
                n_total_count += x.shape[0]
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                predicted = self.model(x)
                loss = self.criterion(predicted, y)
                loss.backward()
                self.optimizer.step()
                n_correct_count += int(torch.sum(torch.argmax(predicted, 1) == y))
                accuracy = n_correct_count / n_total_count 
                tqdm_obj.set_description(desc=return_string("TRAINNING", loss, accuracy))
                if self.logs_path:
                    summary_writer.add_scalar('Loss/train', loss, counter)
                    summary_writer.add_scalar('Accuracy/train', accuracy, counter)

            if self.validation_dataset_loader:
                self.model.eval()
                n_total_count = 0
                n_correct_count = 0
                tqdm_obj = tqdm(self.validation_dataset_loader, ncols=100)
                counter = len(tqdm_obj) * i

                with torch.no_grad():
                    for x, y in tqdm_obj:
                        counter += 1
                        n_total_count += x.shape[0]
                        x, y = x.to(self.device), y.to(self.device)
                        predicted = self.model(x)
                        loss = self.criterion(predicted, y)
                        n_correct_count += int(torch.sum(torch.argmax(predicted, 1) == y))
                        accuracy = n_correct_count / n_total_count
                        tqdm_obj.set_description(desc=return_string("VALIDATION", loss, accuracy))
                        if self.logs_path:
                            summary_writer.add_scalar('Loss/validation', loss, counter)
                            summary_writer.add_scalar('Accuracy/validation', accuracy, counter)

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if self.weights_path:
                torch.save(self.model.state_dict(), self.weights_path.format(epoch=iteration))

class ImageNetBuilder():
    def __init__(self, config):
        self.height = config.IMAGE.HEIGHT
        self.width = config.IMAGE.WIDTH
        self.base_path = config.BACKBONE.DATASET.BASE_PATH
        self.channels = config.IMAGE.CHANNELS
        self.classes = config.BACKBONE.MODEL.CLASSES
        self.batch_size = config.BACKBONE.TRAINING.BATCH_SIZE
        self.model_name = config.BACKBONE.MODEL.NAME
        self.epochs = config.BACKBONE.TRAINING.EPOCHS
        self.optimizer_config = config.BACKBONE.OPTIMIZER
        self.lr_epochs = config.BACKBONE.LEARNING_RATE_SCHEDULE.EPOCHS
        self.weights_path, self.logs_path, metadata_path = create_and_get_paths(config.BACKBONE.TRAINING.SAVE_PATH)
        with open(metadata_path, "w") as outfile:
            json.dump(config, outfile)

    def build(self):
        # Get model
        model = backbones[self.model_name](self.channels, self.classes)

        # Get the training dataset loader
        training_dataset_loader_obj = ImageNetDatasetLoader(self.base_path, self.height, self.width, self.batch_size)

        # Get the validation dataset loader
        # TODO: Add validation dataset loader

        # Create the criterion
        criterion = torch.nn.CrossEntropyLoss()

        # Get the optimizer
        optimizer = get_optimizer(model.parameters(), self.optimizer_config)

        # Create lr_scheduler
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_epochs)

        # Create trainer object
        self.trainer_obj = Trainer(model, optimizer, criterion, self.epochs, training_dataset_loader_obj.get_loader(), lr_scheduler=lr_scheduler, weights_path=self.weights_path, logs_path=self.logs_path)

    def get_trainer(self):
        return self.trainer_obj