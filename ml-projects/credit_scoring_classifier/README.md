# Credit Scoring Classifier Project

A modular, testable, and production-ready machine learning project for credit risk scoring. The goal is to predict whether a customer is likely to default on a loan, using structured financial and behavioral data.

---

## 📂 Project Structure

```
.
├── api/                # REST API with FastAPI
├── data/               # Data loading, preprocessing, feature engineering
├── evaluation/         # Metrics and visualization
├── model/              # Model builders: Logistic Regression, XGBoost, RandomForest
├── observer/           # MLflow integration
├── training/           # Training engine and callbacks
├── ui/                 # Streamlit app for user-friendly prediction interface
├── utils/              # Constants, loggers, helpers
├── tests/              # Unit tests
├── Dockerfile
├── docker-compose.yml
├── main.py             # Training pipeline entry point
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the dataset from Kaggle: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
Place the CSV in the `data/raw/` directory.

### 3. Train the Model

```bash
python main.py
```

Trained model will be saved in `model/best_credit_model.pkl`

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Run Streamlit UI

```bash
streamlit run ui/app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

---

## MLflow Tracking

* MLflow used to track experiments, parameters, and metrics.
* Artifacts stored locally in `mlruns/`

---

## Design Patterns Used

* **Strategy Pattern**: To switch between different scoring models
* **Factory Pattern**: For model instantiation
* **Observer Pattern**: MLflow experiment tracker
* **Builder Pattern**: For preprocessing pipelines

---

## API Usage

### Endpoint: `POST /predict`

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.2,
  "age": 45,
  "NumberOfTime30-59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.8,
  "MonthlyIncome": 4000,
  "NumberOfOpenCreditLinesAndLoans": 4,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}
```

**Response:**

```json
{
  "risk": "Low",
  "score": 0.92
}
```

---

## Features

* Preprocessing pipeline with missing value handling and scaling
* Logistic Regression baseline, and ensemble models (XGBoost, RF)
* API + UI for real-time predictions
* MLflow integration for experiment management

---

## Future Improvements

* Add SHAP-based explainability to the UI
* Integrate with a real-time scoring system
* Support batch predictions from CSV
* Add CI pipeline and model versioning
