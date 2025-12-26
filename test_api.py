# Import packages
from fastapi.testclient import TestClient

from api.main import app

# Test 1
def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"greeting": "Hello World!"}

# Test 2
def test_prediction_negative_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "age": 45,
                "workclass": "Private",
                "fnlgt": 284582,
                "education": "Masters",
                "education-num": 14,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 50,
                "native-country": "United-States",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"prediction": {"salary": " <=50K"}}

# Test 3
def test_prediction_positive_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "age": 31,
                "workclass": "Private",
                "fnlgt": 45781,
                "education": "Masters",
                "education-num": 14,
                "marital-status": "Never-married",
                "occupation": "Prof-specialty",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Female",
                "capital-gain": 14084,
                "capital-loss": 0,
                "hours-per-week": 50,
                "native-country": "United-States",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"prediction": {"salary": " >50K"}}
