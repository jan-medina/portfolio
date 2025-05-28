# evaluation/metrics.py

import torch
import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from utils.logger import get_logger
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

logger = get_logger("Metrics")

@dataclass
class EvaluationConfig:
    model: nn.Module
    dataloader: DataLoader
    device: str = 'cpu'
    class_names: list = None
    verbose: bool = True

def evaluate_classification(config: EvaluationConfig):
    model = config.model.to(config.device)
    dataloader = config.dataloader
    class_names = config.class_names or []
    device = config.device
    verbose = config.verbose
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    num_classes = len(class_names) if class_names else np.max(np.concatenate([y_true, y_pred])) + 1
    labels = list(range(num_classes))

    if verbose:
        logger.info("Classification Report:")
        logger.info(classification_report(y_true, y_pred, target_names=class_names))

        logger.info("Confusion Matrix:")
        logger.info(confusion_matrix(y_true, y_pred))

    return {
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
