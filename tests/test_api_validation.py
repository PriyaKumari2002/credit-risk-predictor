import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_api_health():
    response = client.get("/")
    assert response.status_code == 200
    assert "Credit Risk API" in response.json()["message"]

def test_api_predict_valid_input():
    payload = {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.6,
        "AGE_YEARS": 35,
        "YEARS_EMPLOYED": 5,
        "DEBT_INCOME_RATIO": 0.3,
        "CREDIT_TERM": 60
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "decision" in data
    assert data["decision"] in ["APPROVE", "REJECT"]

def test_api_predict_invalid_input():
    payload = {"EXT_SOURCE_2": "invalid"}
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]

def test_api_predict_missing_fields():
    payload = {"EXT_SOURCE_2": 0.5}
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]