# Put the code for your API here.

from joblib import load
import os
from typing import Dict
from sys import exit
import numpy as np

from fastapi import FastAPI

from api.schemas import PredictPayload
from ml.data import process_data
from ml.model import inference

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

app = FastAPI()


@app.on_event("startup")
def load_artifacts():
    global model
    global encoder
    global lb
    model = load("./model/model.pkl")
    encoder = load("./model/encoder.pkl")
    lb = load("./model/lb.pkl")


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/predict")
async def predict(payload: PredictPayload) -> Dict:
    # Convert payload → dict using original (hyphenated) feature names
    data = payload.dict(by_alias=True)

    # Build continuous feature array (1 row)
    X_continuous = np.array(
        [[data[f] for f in cont_features]],
        dtype=float
    )

    # Build categorical feature array (1 row)
    X_categorical = np.array(
        [[data[f] for f in cat_features]],
        dtype=object
    )

    # Apply trained encoder
    X_cat_encoded = encoder.transform(X_categorical)

    # Concatenate continuous + encoded categorical
    X = np.concatenate([X_continuous, X_cat_encoded], axis=1)
    
    # Run inference
    preds = inference(model, X)
    label = lb.inverse_transform(preds)[0]

    response = {"prediction": {"salary": label}}
    return response