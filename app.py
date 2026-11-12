"""Streamlit dashboard: People Analytics Attrition Insights.

Run:  streamlit run app.py
Shows headcount, turnover rate, attrition by segment, model metrics and the
rule-based insights for HR leaders.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.data import load_data
from src.insights import generate_insights
from src.metrics import compute_metrics
from src.model import train_and_evaluate

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "results.json"


@st.cache_data
def _load_all():
    df = load_data()
    if RESULTS_PATH.exists():
        stored = json.loads(RESULTS_PATH.read_text())
        return df, stored["metrics"], stored["model"], stored["insights"]
    metrics = compute_metrics(df)
    model = train_and_evaluate(df)
    insights = generate_insights(metrics, model)
    return df, metrics, model, insights


def _bar(title, labels, rates):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, [r * 100 for r in rates], color="#2563eb")
    ax.set_title(title)
    ax.set_ylabel("Attrition rate (%)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="People Analytics Attrition Insights", layout="wide")
    st.title("People Analytics Attrition Insights")
    st.caption(
        "Personal project on IBM's published HR Analytics sample data (1,470 employees). "
        "Not Meta data, not real company HR data. Insights are rule-based, not LLM-generated."
    )
    _df, metrics, model, insights = _load_all()

    c1, c2, c3 = st.columns(3)
    c1.metric("Headcount", metrics["headcount"])
    c2.metric("Attrition rate", f"{metrics['attrition_rate'] * 100:.1f}%")
    c3.metric("Employees who left", metrics["attrition_count"])

    st.subheader("Attrition by segment")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(_bar("By department", list(metrics["by_department"]), [v["rate"] for v in metrics["by_department"].values()]))
        st.pyplot(_bar("By overtime", list(metrics["by_overtime"]), [v["rate"] for v in metrics["by_overtime"].values()]))
    with col2:
        st.pyplot(_bar("By tenure band", list(metrics["by_tenure_band"]), [v["rate"] for v in metrics["by_tenure_band"].values()]))
        st.pyplot(_bar("By job role", list(metrics["by_job_role"]), [v["rate"] for v in metrics["by_job_role"].values()]))

    st.subheader("Attrition prediction (held-out test set, 20%)")
    st.write(
        f"Test set: {model['test_size']} employees, {model['test_positives']} leavers. "
        "Stratified 80/20 split, random_state=42."
    )
    st.table(
        {
            "Model": ["Logistic regression", "Random forest"],
            "ROC-AUC": [model["logistic_regression"]["roc_auc"], model["random_forest"]["roc_auc"]],
            "Precision": [model["logistic_regression"]["precision"], model["random_forest"]["precision"]],
            "Recall": [model["logistic_regression"]["recall"], model["random_forest"]["recall"]],
            "F1": [model["logistic_regression"]["f1"], model["random_forest"]["f1"]],
        }
    )
    st.subheader("Top attrition drivers (random forest)")
    st.table(model["top_drivers_random_forest"][:10])

    st.subheader("Insights for HR leaders (rule-based)")
    for item in insights:
        st.markdown(f"**{item['title']}** — {item['finding']}")
        st.markdown(f"*Recommended action:* {item['action']}")


if __name__ == "__main__":
    main()
