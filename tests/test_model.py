"""Model runs, metric ranges, and no-leakage checks."""
from src.data import DROP_COLS
from src.model import build_features


def test_no_nans_in_model_inputs(df):
    X, y = build_features(df)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert len(X) == 1470


def test_leakage_columns_excluded(df):
    X, _y = build_features(df)
    for col in DROP_COLS:
        assert col not in X.columns
    assert "Attrition" not in X.columns


def test_model_metrics_in_valid_ranges(model):
    assert model["test_size"] == 294  # 20% of 1470
    assert model["test_positives"] == 47  # 20% of 237, stratified
    for name in ("logistic_regression", "random_forest"):
        for metric in ("roc_auc", "precision", "recall", "f1"):
            value = model[name][metric]
            assert 0.0 <= value <= 1.0, f"{name}.{metric} = {value}"


def test_models_beat_chance(model):
    assert model["logistic_regression"]["roc_auc"] > 0.5
    assert model["random_forest"]["roc_auc"] > 0.5


def test_top_drivers_present(model):
    drivers = model["top_drivers_random_forest"]
    assert len(drivers) == 10
    assert all(d["importance"] > 0 for d in drivers)
    assert drivers[0]["importance"] >= drivers[-1]["importance"]
