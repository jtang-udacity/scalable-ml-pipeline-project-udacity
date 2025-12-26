# Script to train machine learning model.
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

import joblib

# Add the necessary imports for the starter code.

from ml.data import process_data
from ml.model import compute_model_metrics, inference 

# Function to compute model metrics on data slices
def compute_model_metrics_slice(model, data, encoder, lb, cat_features, sliced_feature, label):

    dict_result = {}

    for i in data[sliced_feature].unique():
        data_slice = data[data[sliced_feature] == i]

        X, y, _, _ = process_data(
            data_slice,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)

        dict_result[i] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "sample": len(y),
        }

    return {sliced_feature: dict_result}

# Add code to load in the data.

data_loc = "./data/census.csv"
data = pd.read_csv(data_loc)
data.columns = data.columns.str.strip()

# Code to load in model artifacts

model = joblib.load("./model/model.pkl")
encoder = joblib.load("./model/encoder.pkl")
lb = joblib.load("./model/lb.pkl")

# Split train and test data
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

# Compute metrics on metric slices
metrics_education = compute_model_metrics_slice(model, test, encoder, lb, cat_features, "education", "salary")

with open("./screenshots/slice_output.json", "w") as fp:
    json.dump(metrics_education, fp)