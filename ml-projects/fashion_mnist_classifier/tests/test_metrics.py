import torch
from model.cnn_model import build_model
from evaluation.metrics import evaluate_classification, EvaluationConfig
from torch.utils.data import DataLoader, TensorDataset

def test_evaluate_classification_runs():
    model = build_model()
    dummy_data = torch.randn(20, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (20,))
    ds = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(ds, batch_size=4)
    class_names = [str(i) for i in range(10)]

    config = EvaluationConfig(model=model, dataloader=dataloader, device='cpu', class_names=class_names, verbose=False)
    result = evaluate_classification(config)

    assert "confusion_matrix" in result
    assert "classification_report" in result
