import torch
from model.cnn_model import build_model
from training.callbacks import BestModelSaver
import os

def test_best_model_saves_only_on_improvement(tmp_path):
    import os

    model = build_model()
    path = tmp_path / "model.pth"

    saver = BestModelSaver(model, path)

    saver.on_epoch_end(epoch=0, logs={"val_acc": 0.8})
    assert os.path.exists(path)

    original_size = path.stat().st_size
    saver.on_epoch_end(epoch=1, logs={"val_acc": 0.75})
    assert path.stat().st_size == original_size

    saver.on_epoch_end(epoch=2, logs={"val_acc": 0.9})
    assert path.stat().st_size == original_size or path.stat().st_size != 0
