# evaluation/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dataclasses import dataclass
import io

@dataclass
class ConfusionMatrixPlotConfig:
    cm: any
    class_names: list
    normalize: bool = False
    title: str = "Confusion Matrix"
    figsize: tuple = (8, 6)


def plot_confusion_matrix(config: ConfusionMatrixPlotConfig):
    """
    Dibuja una matriz de confusión.
    """
    cm = config.cm
    if cm.normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=cm.figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=cm.class_names, yticklabels=cm.class_names)
    plt.title(cm.title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels, class_names):
    """
    Muestra la distribución de clases en un conjunto de etiquetas.
    """
    from collections import Counter
    counts = Counter(labels)
    sorted_labels = [class_names[i] for i in range(len(class_names))]

    plt.figure(figsize=(8, 4))
    sns.barplot(x=sorted_labels, y=[counts[i] for i in range(len(class_names))])
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def render_confusion_matrix_image(config: ConfusionMatrixPlotConfig):
    """
    Genera un PNG de la matriz de confusión en un buffer en memoria.
    """
    cm = config.cm
    if config.normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=config.figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=config.class_names, yticklabels=config.class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(config.title)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf