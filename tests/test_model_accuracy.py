import pytest
import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "src" / "model.pkl"
FEATURES_PATH = Path(__file__).parent.parent / "src" / "feature_names.pkl"

@pytest.fixture
def model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@pytest.fixture
def feature_names():
    with open(FEATURES_PATH, 'rb') as f:
        return pickle.load(f)

def test_model_prediction_output_format(model, feature_names):
    dummy_input = np.random.rand(10, len(feature_names))
    predictions = model.predict(dummy_input)
    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)

def test_model_predict_proba_output(model, feature_names):
    dummy_input = np.random.rand(5, len(feature_names))
    probabilities = model.predict_proba(dummy_input)
    assert probabilities.shape[0] == 5
    assert probabilities.shape[1] == 2
    assert all(0 <= p <= 1 for row in probabilities for p in row)

def test_model_handles_edge_cases(model, feature_names):
    # Zero values
    zero_input = np.zeros((1, len(feature_names)))
    pred_zero = model.predict(zero_input)
    assert pred_zero[0] in [0, 1]
    
    # Large values
    large_input = np.full((1, len(feature_names)), 1000)
    pred_large = model.predict(large_input)
    assert pred_large[0] in [0, 1]

def test_model_consistency(model, feature_names):
    same_input = np.array([[0.5] * len(feature_names)])
    pred1 = model.predict(same_input)
    pred2 = model.predict(same_input)
    assert pred1[0] == pred2[0], "Model predictions inconsistent"