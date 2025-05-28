from training.callbacks import PrintLogger

def test_print_logger_does_not_fail(capsys):
    logger = PrintLogger()
    logger.on_epoch_end(epoch=0, logs={"loss": 0.45, "val_acc": 0.88})

    captured = capsys.readouterr()
    assert "[Epoch 1]" in captured.out
    assert "Loss: 0.45" in captured.out
    assert "Val Acc: 0.88" in captured.out
