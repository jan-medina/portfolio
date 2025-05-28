# FashionMNIST Classifier Project

A modular, testable, and extensible machine learning project for image classification using the FashionMNIST dataset. The project includes data loading, model building, training pipeline, metrics evaluation, visualization tools, a REST API using FastAPI, and a Streamlit-based user interface.

---

## Project Structure

```
.
├── api/                # REST API with FastAPI
├── data/               # Data loading utilities and dataset handling
├── evaluation/         # Metrics and visualization
├── model/              # CNN models and architecture variations
├── observer/           # MLflow adapter for experiment tracking
├── tests/              # Unit tests and integration tests
├── training/           # Training engine and callbacks
├── ui/                 # Streamlit user interface
├── utils/              # Logging, constants, and helper functions
├── Dockerfile
├── docker-compose.yml
├── main.py             # Training entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/your-user/fashionmnist-classifier.git
cd fashionmnist-classifier
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py
```

Trained model will be saved as `model/best_fashion.pth`.

### 3. Run the API

```bash
uvicorn api.main:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run the UI

```bash
streamlit run ui/app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

### 5. Run Tests

```bash
pytest
```

---

## Docker Support

### Build and Run with Docker Compose

```bash
docker-compose up --build
```

* FastAPI will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)
* Streamlit UI will be available at: [http://localhost:8501](http://localhost:8501)

---

## API Usage

### Endpoint: `POST /predict`

Accepts an image file and returns predicted class.

**Request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -F 'file=@path_to_image.jpg'
```

**Response:**

```json
{
  "predicted_class": "Ankle Boot",
  "confidence": 0.9821
}
```

---

## MLflow Tracking

* Automatically logs metrics, parameters, and model artifacts.
* Set up by the `observer/` module via adapter pattern.
* Default MLflow artifacts are saved under `mlruns/`.

---

## Design Patterns Used

* **Strategy Pattern**: for flexible model definitions
* **Observer Pattern**: for MLflow experiment tracking
* **Adapter Pattern**: to plug MLflow into the training engine
* **Factory Pattern**: for CNN model instantiation

---

## Features

* Modular CNN training pipeline
* Reusable engine for other vision tasks
* Metrics tracking (accuracy, F1-score, confusion matrix)
* Streamlit interface for image prediction
* REST API endpoint `/predict`

---

## Evaluation Example

* Visualize confusion matrix with `plot_confusion_matrix()`
* Get model-wide metrics via `evaluate_classification()`

---

## Future Improvements

* Add data augmentation pipeline
* Add support for ONNX export
* Enable model comparison with MLflow UI
* Add CI/CD GitHub Actions workflow
