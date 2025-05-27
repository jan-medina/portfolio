# data/data_loader.py

from abc import ABC, abstractmethod
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class DatasetFactory(ABC):
    @abstractmethod
    def create_dataloaders(self, batch_size: int, val_split: float = 0.1):
        pass


class FashionMNISTFactory(DatasetFactory):
    def create_dataloaders(self, batch_size: int, val_split: float = 0.1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        val_size = int(len(full_train) * val_split)
        train_size = len(full_train) - val_size

        train, val = random_split(full_train, [train_size, val_size])

        return {
            'train': DataLoader(train, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val, batch_size=batch_size),
            'test': DataLoader(test, batch_size=batch_size)
        }


class MNISTFactory(DatasetFactory):
    def create_dataloaders(self, batch_size: int, val_split: float = 0.1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        val_size = int(len(full_train) * val_split)
        train_size = len(full_train) - val_size

        train, val = random_split(full_train, [train_size, val_size])

        return {
            'train': DataLoader(train, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val, batch_size=batch_size),
            'test': DataLoader(test, batch_size=batch_size)
        }
