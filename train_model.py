# Script to train machine learning model.
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

import joblib

# Add the necessary imports for the starter code.

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference 

# Add code to load in the data.

data_loc = "./data/census.csv"
data = pd.read_csv(data_loc)
data.columns = data.columns.str.strip()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
rfc = train_model(X_train, y_train)
rfc_best = rfc.best_estimator_
y_pred = inference(rfc_best, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

# Save model and encoder
output_loc = "../model/"
model_loc = output_loc + "model.pkl"
encoder_loc = output_loc + "encoder.pkl"
lb_loc = output_loc + "lb.pkl"

joblib.dump(rfc_best, model_loc)
joblib.dump(encoder, encoder_loc)
joblib.dump(lb, lb_loc)
