from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
instrumentator = Instrumentator()

instrumentator.instrument(app).expose(app) 

# Load model
model = joblib.load("api/models/naive_bayes.pkl")

class CancerInput(BaseModel):
    features: list[float]  # 30 features

   

@app.get("/")
def home():
    return {"message": "Breast Cancer Diagnosis API is live!"}

@app.post("/predict")
def predict(data: CancerInput):
    input_data = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return {"prediction": int(prediction)}



