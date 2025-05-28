# model/builder.py

from abc import ABC, abstractmethod
import torch.nn as nn


class ModelBuilder(ABC):
    @abstractmethod
    def add_input_layer(self):
        pass

    @abstractmethod
    def add_conv_layers(self):
        pass

    @abstractmethod
    def add_fc_layers(self):
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass


class CNNClassifierBuilder(ModelBuilder):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        self.model = nn.Sequential()
        self.input_shape = input_shape
        self.num_classes = num_classes

    def add_input_layer(self):
        # No need for explicit input layer in PyTorch Sequential
        return self

    def add_conv_layers(self):
        self.model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('pool1', nn.MaxPool2d(2))

        self.model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('pool2', nn.MaxPool2d(2))

        return self

    def add_fc_layers(self):
        self.model.add_module('flatten', nn.Flatten())
        self.model.add_module('fc1', nn.Linear(64 * 7 * 7, 128))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(128, self.num_classes))

        return self

    def get_model(self):
        return self.model

class CNNClassifierBuilderV2(ModelBuilder):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, use_dropout=True, use_batchnorm=True):
        self.model = nn.Sequential()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

    def add_input_layer(self):
        return self

    def add_conv_layers(self):
        self.model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
        if self.use_batchnorm:
            self.model.add_module('bn1', nn.BatchNorm2d(32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('pool1', nn.MaxPool2d(2))

        self.model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
        if self.use_batchnorm:
            self.model.add_module('bn2', nn.BatchNorm2d(64))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('pool2', nn.MaxPool2d(2))

        return self

    def add_fc_layers(self):
        self.model.add_module('flatten', nn.Flatten())
        self.model.add_module('fc1', nn.Linear(64 * 7 * 7, 128))
        if self.use_dropout:
            self.model.add_module('dropout1', nn.Dropout(0.5))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(128, self.num_classes))
        return self

    def get_model(self):
        return self.model
