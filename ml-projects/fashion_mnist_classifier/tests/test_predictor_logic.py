import torch
import numpy as np
from prediction.predictor import Predictor
from model.cnn_model import build_model
from PIL import Image

def test_predictor_accepts_tensor_and_array():
    model = build_model()
    model.eval()

    predictor = Predictor(model)

    dummy_tensor = torch.rand(28, 28)  # grayscale
    dummy_array = (np.random.rand(28, 28) * 255).astype(np.uint8)
    dummy_image = Image.fromarray(dummy_array)

    for input_data in [dummy_tensor, dummy_array, dummy_image]:
        pred = predictor.predict(input_data)
        assert isinstance(pred, int)
