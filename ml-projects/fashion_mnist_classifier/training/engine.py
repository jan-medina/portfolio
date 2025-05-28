# training/engine.py

from abc import ABC
import torch

class TrainingEngine(ABC):
    def __init__(self, model, dataloaders, device='cpu', callbacks=None):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.criterion = self.define_criterion()
        self.optimizer = self.define_optimizer()
        self.callbacks = callbacks or []

    def notify(self, hook, **kwargs):
        for cb in self.callbacks:
            getattr(cb, hook)(**kwargs)

    def train(self, num_epochs=5):
        self.notify('on_train_start')

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.dataloaders['train']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.dataloaders['train'])
            val_acc = self.validate()

            self.notify('on_epoch_end', epoch=epoch, logs={'loss': avg_loss, 'val_acc': val_acc})

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.dataloaders['val']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
