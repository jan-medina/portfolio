import yaml
import torch

class Config:
    def __init__(self, path='config.yml'):
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        self.batch_size = raw.get('batch_size', 64)
        self.val_split = raw.get('val_split', 0.1)
        self.learning_rate = raw.get('learning_rate', 0.001)
        self.epochs = raw.get('epochs', 5)

        device = raw.get('device', 'cpu')
        self.device = (
            'cuda' if device == 'auto' and torch.cuda.is_available()
            else 'cpu' if device == 'auto'
            else device
        )

        self.use_dropout = raw.get('use_dropout', True)
        self.use_batchnorm = raw.get('use_batchnorm', True)
        self.model_save_path = raw.get('model_save_path', 'model/best_fashion.pth')

    def as_dict(self):
        return self.__dict__

config = Config()
