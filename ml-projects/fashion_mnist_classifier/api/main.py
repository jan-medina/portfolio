from fastapi import FastAPI, UploadFile, File, HTTPException
from evaluation.metrics import evaluate_classification, EvaluationConfig
from evaluation.visualizer import render_confusion_matrix_image, ConfusionMatrixPlotConfig
from loader.data_loader import FashionMNISTFactory
from fastapi.responses import StreamingResponse
from prediction.predictor import Predictor
from model.cnn_model import build_model
from utils.logger import get_logger
from PIL import Image
import torch
import io

app = FastAPI(title="FashionMNIST Classifier API")
logger = get_logger("predict")

# Load model on startup
model = build_model(version='v2', use_dropout=True, use_batchnorm=True)
model.load_state_dict(torch.load("model/best_fashion.pth", map_location="cpu"))
predictor = Predictor(model)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        class_index = predictor.predict(image)
        return {
            "predicted_class": class_names[class_index],
            "class_index": class_index
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    dataloaders = FashionMNISTFactory().create_dataloaders(batch_size=64)
    eval_config = EvaluationConfig(
        model=predictor.model,
        dataloader=dataloaders["test"],
        device="cpu",
        class_names=class_names,
        verbose=False
    )

    results = evaluate_classification(eval_config)

    return {
        "accuracy": results["classification_report"]["accuracy"],
        "f1_macro": results["classification_report"]["macro avg"]["f1-score"],
        "confusion_matrix": results["confusion_matrix"].tolist()
    }

@app.get("/metrics/plot")
def get_confusion_matrix_plot():
    dataloaders = FashionMNISTFactory().create_dataloaders(batch_size=64)
    eval_config = EvaluationConfig(
        model=predictor.model,
        dataloader=dataloaders["test"],
        device="cpu",
        class_names=class_names,
        verbose=False
    )

    results = evaluate_classification(eval_config)

    img_buffer = render_confusion_matrix_image(ConfusionMatrixPlotConfig(
        cm=results["confusion_matrix"],
        class_names=class_names,
        normalize=True,
        title="Confusion Matrix (Normalized)"
    ))

    return StreamingResponse(img_buffer, media_type="image/png")