# prediction/predictor.py

import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class Predictor:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, input_data):
        tensor = self._adapt_input(input_data).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            _, predicted = torch.max(output, 1)
        return predicted.item()

    def _adapt_input(self, input_data):
        if isinstance(input_data, Image.Image):
            return self.transform(input_data)
        elif isinstance(input_data, np.ndarray):
            img = Image.fromarray(input_data.astype(np.uint8))
            return self.transform(img)
        elif isinstance(input_data, torch.Tensor):
            if input_data.ndim == 2:
                input_data = input_data.unsqueeze(0)
            return self.transform(transforms.ToPILImage()(input_data))
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")
