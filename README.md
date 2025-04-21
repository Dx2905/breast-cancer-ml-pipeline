# 🧠 Breast Cancer Diagnosis using Machine Learning

## 📌 Project Overview
This project focuses on predicting breast cancer diagnosis (Benign vs Malignant) using a **robust ML pipeline**. It combines **traditional machine learning models** with modern **MLOps tooling** for reproducibility, explainability, and deployment.

> ✅ Built in alignment with resume claims, including: SHAP, MLflow, FastAPI, Prometheus, and Airflow integration.

---

## 🗂 Dataset
- **Source**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Samples**: 569
- **Classes**: Malignant (M), Benign (B)
- **Features**: 30 numerical attributes describing cell nuclei

---

## 🧪 Machine Learning Models
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM - Linear & RBF)**
- **Naive Bayes**
- **Decision Tree**
- **Random Forest**

### ✅ Techniques Used
- **10-Fold Cross Validation**
- **Hyperparameter tuning**: `GridSearchCV`, `RandomizedSearchCV`
- **Preprocessing**: StandardScaler, Label Encoding

---

## 🪄 Project Features

### 🔍 1. Model Training
All models are trained on normalized and label-encoded features. The best-performing models are saved with `joblib`.

### 📈 2. MLflow Tracking
Each model run is logged with:
- Parameters
- Accuracy, Recall, F1-score
- Confusion Matrix (as JSON)
- Full model artifact logging (using `mlflow.sklearn.log_model()`)

### 🧠 3. SHAP Explainability
SHAP values were generated and visualized for:
- **Logistic Regression**
- **SVM (Linear)**

Plots include:
- SHAP summary plot
- SHAP bar plot
Saved in: `plots/` directory.

### 🌐 4. FastAPI Model Serving
- RESTful API created with **FastAPI**
- Load best model and expose `/predict` endpoint
- Test with `predict.py` or via Swagger UI at `localhost:8000/docs`

### 📊 5. Prometheus Monitoring
- Request counts, response size, and durations tracked
- Accessible at: `localhost:8000/metrics`
- Integrated with **Grafana dashboard** (see JSON in `monitoring/`)

### ⏱ 6. Airflow Retraining DAG
- **airflow/dags/breast_cancer_retrain_dag.py** schedules weekly retraining
- Runs `train.py` every week
- Uses BashOperator (can extend to data fetching or uploading)

---

## 🧪 Metrics Snapshot

| Model              | Accuracy | Recall  | F1 Score |
|-------------------|----------|---------|----------|
| Logistic Regression | 93.9%    | 97.7%   | 92.3%    |
| KNN                | 94.7%    | 93.0%   | 93.0%    |
| SVM (Linear)       | 95.6%    | 95.3%   | 94.2%    |
| Random Forest      | 96.5%    | 93.0%   | 95.2%    |

> Detailed logs and plots saved in `mlruns/`, `plots/`, and `airflow/logs/`.

---

## 🚀 Getting Started

### 🔧 Installation
```bash
git clone https://github.com/Dx2905/breast-cancer-ml-pipeline.git
cd breast-cancer-ml-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 🏃 Run Training
```bash
python train.py
```

### 🧪 Test API
```bash
uvicorn app:app --reload
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample.json
```

### 📈 Launch Prometheus
```bash
./prometheus --config.file=prometheus.yml
```

### 📊 Launch Airflow
```bash
airflow standalone
```

---

## 📁 Project Structure
```
.
├── train.py
├── mlflow_utils.py
├── explainability.py
├── app.py
├── api/
├── airflow/
│   └── dags/
├── models/
├── plots/
└── data/
    └── wdbc.csv
```

---

## 💡 Future Improvements
- Dockerize full stack (FastAPI + Prometheus + Airflow)
- Add LIME explainability
- Integrate CI/CD (GitHub Actions)
- Auto-retrain with new dataset via Airflow sensor

---

## 📜 License
This project is under the [MIT License](LICENSE).
