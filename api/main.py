# Put the code for your API here.

from joblib import load
import os
from typing import Dict
from sys import exit

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
    import pandas as pd
    data = pd.DataFrame(payload.dict(by_alias=True), index=[0])
    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X)
    label = lb.inverse_transform(preds)[0]

    response = {"prediction": {"salary": label}}
    return response