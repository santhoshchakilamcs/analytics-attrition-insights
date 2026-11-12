"""Run metrics + model end to end; write results/results.json and charts/*.png."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import load_data
from src.insights import generate_insights
from src.metrics import compute_metrics
from src.model import train_and_evaluate

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "results.json"
CHARTS_DIR = ROOT / "charts"


def _bar_chart(title: str, labels: list[str], rates: list[float], filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, [r * 100 for r in rates], color="#2563eb")
    ax.set_title(title)
    ax.set_ylabel("Attrition rate (%)")
    ax.set_ylim(0, max(rates) * 100 * 1.25 if rates else 10)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{rate * 100:.1f}%", ha="center", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150)
    plt.close(fig)


def _driver_chart(drivers: list[dict], filename: str) -> None:
    names = [d["feature"] for d in drivers][::-1]
    values = [d["importance"] for d in drivers][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, values, color="#059669")
    ax.set_title("Top 10 attrition drivers (random forest feature importance)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data()
    metrics = compute_metrics(df)
    model = train_and_evaluate(df)
    insights = generate_insights(metrics, model)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {"metrics": metrics, "model": model, "insights": insights}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {RESULTS_PATH}")

    _bar_chart(
        "Attrition rate by department",
        list(metrics["by_department"].keys()),
        [v["rate"] for v in metrics["by_department"].values()],
        "attrition_by_department.png",
    )
    _bar_chart(
        "Attrition rate by overtime",
        list(metrics["by_overtime"].keys()),
        [v["rate"] for v in metrics["by_overtime"].values()],
        "attrition_by_overtime.png",
    )
    _bar_chart(
        "Attrition rate by tenure band",
        list(metrics["by_tenure_band"].keys()),
        [v["rate"] for v in metrics["by_tenure_band"].values()],
        "attrition_by_tenure_band.png",
    )
    _bar_chart(
        "Attrition rate by job role",
        list(metrics["by_job_role"].keys()),
        [v["rate"] for v in metrics["by_job_role"].values()],
        "attrition_by_job_role.png",
    )
    _driver_chart(model["top_drivers_random_forest"], "top_drivers.png")
    print(f"Wrote charts to {CHARTS_DIR}")

    print(f"Headcount: {metrics['headcount']}")
    print(f"Overall attrition rate: {metrics['attrition_rate'] * 100:.1f}%")
    print(f"Random forest test ROC-AUC: {model['random_forest']['roc_auc']}")
    print(f"Logistic regression test ROC-AUC: {model['logistic_regression']['roc_auc']}")


if __name__ == "__main__":
    main()
