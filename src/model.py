"""Attrition prediction on the IBM HR Analytics dataset.

Stratified 80/20 train/test split (random_state=42 throughout, so results are
stable across runs). Two baselines: logistic regression and random forest.
No target leakage: EmployeeNumber / EmployeeCount / StandardHours / Over18 are
excluded (ID or constants); the target column is never a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import DROP_COLS, TARGET

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categoricals; return (X, y) with no NaNs."""
    work = df.drop(columns=DROP_COLS)
    y = (work[TARGET] == "Yes").astype(int)
    X = work.drop(columns=[TARGET])
    X = pd.get_dummies(X, drop_first=False)
    X = X.astype(float)
    return X, y


def _score(y_true, y_pred, y_proba) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train both models on a stratified split; return test metrics + drivers."""
    X, y = build_features(df)
    if X.isna().any().any():
        raise ValueError("NaNs found in model features")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    lr = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_metrics = _score(y_test, lr_pred, lr_proba)

    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = _score(y_test, rf_pred, rf_proba)

    importances = sorted(
        zip(X.columns, rf.feature_importances_), key=lambda pair: pair[1], reverse=True
    )
    top_drivers = [
        {"feature": name, "importance": round(float(value), 4)}
        for name, value in importances[:10]
    ]

    coefs = lr.named_steps["clf"].coef_[0]
    lr_top = sorted(zip(X.columns, coefs), key=lambda pair: abs(pair[1]), reverse=True)[:10]
    lr_drivers = [
        {"feature": name, "coefficient": round(float(value), 4)} for name, value in lr_top
    ]

    return {
        "test_size": int(len(y_test)),
        "test_positives": int(y_test.sum()),
        "train_size": int(len(y_train)),
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "top_drivers_random_forest": top_drivers,
        "top_drivers_logistic_regression": lr_drivers,
    }
