"""Shared fixtures for tests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.metrics import compute_metrics
from src.model import build_features, train_and_evaluate


@pytest.fixture(scope="session")
def df():
    return load_data()


@pytest.fixture(scope="session")
def metrics(df):
    return compute_metrics(df)


@pytest.fixture(scope="session")
def model(df):
    return train_and_evaluate(df)
