'''
Test for train_model.py (Model Training)

Author: Jason
'''

# import libraries
import joblib
import numpy as np
import pandas as pd
import pytest 
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, inference

# Pytest fixtures
@pytest.fixture
def data():
    df = pd.read_csv("../data/census.csv")
    return df

@pytest.fixture
def encoder():
    encoder = joblib.load("../model/encoder.pkl")
    return encoder

@pytest.fixture
def lb():
    lb = joblib.load("../model/lb.pkl")
    return lb

@pytest.fixture
def model():
    model = joblib.load("../model/model.pkl")
    return model

# Test 1
'''
Test to confirm data loaded successfully 
'''
def test_data_shape(data):
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    
# Test 2
'''
Test to confirm data processed succesfully 
'''
def test_process_data(data, encoder, lb):
    
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
    
    data.columns = data.columns.str.strip()
    
    X, y, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    ) 

    assert data.shape[0] == np.shape(X)[0]
    assert data.shape[0] == np.shape(y)[0]
    assert data.shape[1] <= np.shape(X)[1]
    assert np.ndim(y) == 1

# Test 3
'''
Test to confirm model output meets minimal performance requirements 
'''
def test_model(data, model, encoder, lb):

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
    
    data.columns = data.columns.str.strip()

    train, test = train_test_split(data, test_size=0.20, random_state=42)
    
    X, y, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    ) 

    y_pred = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, y_pred)

    assert fbeta >= 0.5 
