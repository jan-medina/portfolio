# main.py

import torch
from loader.data_loader import FashionMNISTFactory
from model.cnn_model import build_model
from training.fashion_trainer import FashionTrainer
from training.callbacks import PrintLogger, BestModelSaver, MLflowTracker
from utils.config import config
from utils.logger import get_logger

logger = get_logger("main")

def main():
    logger.info("🚀 Starting FashionMNIST training pipeline")

    dataloaders = FashionMNISTFactory().create_dataloaders(
        batch_size=config.batch_size,
        val_split=config.val_split
    )

    logger.info("📦 Dataloaders created")

    model = build_model(
        version='v2',
        use_dropout=config.use_dropout,
        use_batchnorm=config.use_batchnorm
    )

    logger.info("🧠 Model built")

    callbacks = [
        PrintLogger(),
        BestModelSaver(model, path=config.model_save_path),
        MLflowTracker(model, experiment_name="FashionMNIST", params=config.as_dict())
    ]

    trainer = FashionTrainer(model, dataloaders, config.device, callbacks=callbacks)
    trainer.train(num_epochs=config.epochs)

    logger.info("✅ Training completed")


if __name__ == '__main__':
    main()
