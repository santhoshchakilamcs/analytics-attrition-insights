"""HR metrics computed from the IBM HR Analytics Employee Attrition dataset.

All numbers come from the real (sample) data file; nothing is hard-coded.
Attrition rate = share of employees with Attrition == "Yes".
"""
from __future__ import annotations

import pandas as pd

TENURE_BANDS = [(0, 2, "0-2 yrs"), (3, 5, "3-5 yrs"), (6, 10, "6-10 yrs"), (11, 99, "11+ yrs")]


def _segment(df: pd.DataFrame, column: str) -> dict:
    out = {}
    for name, group in df.groupby(column, observed=True):
        n = len(group)
        left = int((group["Attrition"] == "Yes").sum())
        out[str(name)] = {"n": n, "attrition": left, "rate": round(left / n, 4) if n else 0.0}
    return out


def _tenure_band(years: int) -> str:
    for lo, hi, label in TENURE_BANDS:
        if lo <= years <= hi:
            return label
    return "11+ yrs"


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute headcount, turnover and segment attrition from the data."""
    headcount = len(df)
    left = int((df["Attrition"] == "Yes").sum())
    stayed = headcount - left
    attrition_rate = round(left / headcount, 4) if headcount else 0.0

    leavers = df[df["Attrition"] == "Yes"]
    stayers = df[df["Attrition"] == "No"]

    def _mean(frame: pd.DataFrame, col: str) -> float:
        return round(float(frame[col].mean()), 2)

    tenure = df["YearsAtCompany"].apply(_tenure_band)
    by_tenure = {}
    for label, _lo, _hi in [(b[2], b[0], b[1]) for b in TENURE_BANDS]:
        group = df[tenure == label]
        n = len(group)
        lft = int((group["Attrition"] == "Yes").sum())
        by_tenure[label] = {"n": n, "attrition": lft, "rate": round(lft / n, 4) if n else 0.0}

    return {
        "headcount": headcount,
        "attrition_count": left,
        "stayed_count": stayed,
        "attrition_rate": attrition_rate,
        "by_department": _segment(df, "Department"),
        "by_job_role": _segment(df, "JobRole"),
        "by_overtime": _segment(df, "OverTime"),
        "by_tenure_band": by_tenure,
        "by_business_travel": _segment(df, "BusinessTravel"),
        "leavers_vs_stayers": {
            "leavers": {
                "n": left,
                "avg_tenure_years": _mean(leavers, "YearsAtCompany"),
                "avg_monthly_income": _mean(leavers, "MonthlyIncome"),
                "avg_age": _mean(leavers, "Age"),
                "avg_job_satisfaction": _mean(leavers, "JobSatisfaction"),
                "avg_environment_satisfaction": _mean(leavers, "EnvironmentSatisfaction"),
            },
            "stayers": {
                "n": stayed,
                "avg_tenure_years": _mean(stayers, "YearsAtCompany"),
                "avg_monthly_income": _mean(stayers, "MonthlyIncome"),
                "avg_age": _mean(stayers, "Age"),
                "avg_job_satisfaction": _mean(stayers, "JobSatisfaction"),
                "avg_environment_satisfaction": _mean(stayers, "EnvironmentSatisfaction"),
            },
        },
    }
