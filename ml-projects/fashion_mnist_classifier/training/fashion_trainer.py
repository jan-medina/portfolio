from training.engine import TrainingEngine
import torch.nn as nn
import torch.optim as optim
from utils.config import config


class FashionTrainer(TrainingEngine):
    def define_criterion(self):
        return nn.CrossEntropyLoss()

    def define_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=config.learning_rate)
