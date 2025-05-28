# training/callbacks.py

from abc import ABC
import torch
import mlflow
import mlflow.pytorch

class Callback(ABC):
    def on_train_start(self, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass


class PrintLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"[Epoch {epoch+1}] Loss: {logs['loss']:.4f} - Val Acc: {logs['val_acc']:.4f}")


class BestModelSaver(Callback):
    def __init__(self, model, path='best_model.pth'):
        self.model = model
        self.path = path
        self.best_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_acc', 0.0)
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            torch.save(self.model.state_dict(), self.path)
            print(f"✅ New best model saved at epoch {epoch+1} with val_acc: {val_acc:.4f}")

class MLflowTracker(Callback):
    def __init__(self, model, experiment_name="FashionMNIST", params=None):
        self.model = model
        self.experiment_name = experiment_name
        self.params = params or {}

        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

        # Registrar parámetros del experimento
        for k, v in self.params.items():
            mlflow.log_param(k, v)

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric("loss", logs['loss'], step=epoch)
        mlflow.log_metric("val_accuracy", logs['val_acc'], step=epoch)

    def on_train_start(self, logs=None):
        pass

    def on_train_end(self):
        # Guardar el modelo al finalizar
        mlflow.pytorch.log_model(self.model, artifact_path="model")
        mlflow.end_run()