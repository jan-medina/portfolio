# 🌟 FashionMNIST Classifier - ML Engineering Project

A modular, extensible image classifier built with PyTorch for FashionMNIST, designed with software engineering best practices: design patterns (Builder, Factory, Template, Observer, Adapter), MLOps integrations, and experiment tracking with MLflow.

---

## 📚 Project Structure

```text
fashion_mnist_classifier/
├── data/                  # Data loading (Factory Method)
├── model/                 # Model builders (Builder Pattern)
├── training/              # Training engine + callbacks (Template + Observer)
├── evaluation/            # Metrics & visualizations
├── prediction/            # Unified predictor interface (Adapter Pattern)
├── utils/                 # Config, logger
├── notebooks/             # Optional EDA & experiment logs
├── main.py                # Entrypoint for training
├── requirements.txt       # Dependencies
└── README.md
```

---

## 🤖 Features

### Design Patterns Implemented

* **Factory Method**: Loaders for different datasets (e.g., MNIST, FashionMNIST)
* **Builder**: Flexible CNN architecture with options (dropout, batchnorm)
* **Template Method**: TrainingEngine base class with overridable hooks
* **Observer**: Callbacks for logging, saving models, MLflow integration
* **Adapter**: `Predictor` interface for image/array/tensor inputs

### MLOps Integrations

* MLflow for parameter tracking, metric logging, model artifact storage
* Config management and logging utilities
* Ready for dockerization and deployment

---

## 🔧 Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python main.py
```

### Launch MLflow UI (optional)

```bash
mlflow ui
# Visit http://localhost:5000
```

---

## 🔬 Evaluate Model

```python
from evaluation.metrics import evaluate_classification
from evaluation.visualizer import plot_confusion_matrix

results = evaluate_classification(model, dataloaders['test'], class_names=class_names)
plot_confusion_matrix(results['confusion_matrix'], class_names)
```

---

## 📊 Predict on new image

```python
from prediction.predictor import Predictor
from PIL import Image

model = build_model(version='v2', use_dropout=True, use_batchnorm=True)
model.load_state_dict(torch.load("model/best_fashion.pth"))
predictor = Predictor(model)
img = Image.open("some_test_image.png")
label = predictor.predict(img)
print(f"Predicted: {label}")
```

---

## 🎓 Educational Value

This project is ideal for demonstrating:

* Clean architecture and maintainability in ML systems
* Applying software engineering patterns to ML pipelines
* Reproducibility and experiment management

---

## 🚀 Next Steps

* Add more datasets (e.g., CIFAR10)
* Deploy with FastAPI or Streamlit
* Add pytest unit tests
* Add hyperparameter tuning integration (Optuna)
