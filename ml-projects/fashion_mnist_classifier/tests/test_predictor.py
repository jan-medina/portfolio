import torch
from model.cnn_model import build_model
from prediction.predictor import Predictor
import numpy as np
from PIL import Image

def test_predict_numpy_input():
    model = build_model()
    predictor = Predictor(model)
    dummy_array = (np.random.rand(28, 28) * 255).astype(np.uint8)
    pred = predictor.predict(dummy_array)
    assert isinstance(pred, int)
