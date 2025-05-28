import torch
from model.cnn_model import build_model

def test_build_model_output_shape():
    model = build_model(version='v2', use_dropout=True, use_batchnorm=True)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10)
