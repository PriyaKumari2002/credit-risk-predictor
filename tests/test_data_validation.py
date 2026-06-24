import pytest
import pickle
import pandas as pd
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

def test_model_loads(model):
    assert model is not None

def test_feature_names_load(feature_names):
    assert feature_names is not None
    assert isinstance(feature_names, list)

def test_model_has_predict(model):
    assert hasattr(model, 'predict')

def test_model_prediction(model, feature_names):
    dummy_input = np.random.rand(1, len(feature_names))
    prediction = model.predict(dummy_input)
    assert prediction.shape[0] == 1