"""Deterministic, rule-based insights generator for HR leaders.

No LLM and no API key are involved: each finding is produced by a fixed rule
applied to the computed metrics (e.g. a segment whose attrition rate is more
than 1.25x the overall rate is flagged as elevated). The README labels this
as rule-based, never as LLM-generated.
"""
from __future__ import annotations


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def generate_insights(metrics: dict, model: dict) -> list[dict]:
    """Turn computed metrics + model results into findings and actions."""
    insights: list[dict] = []
    overall = metrics["attrition_rate"]
    headcount = metrics["headcount"]
    left = metrics["attrition_count"]

    insights.append(
        {
            "title": "Overall turnover",
            "finding": (
                f"{left} of {headcount} employees left ({_pct(overall)} attrition). "
                f"{'This is above the 10% level many HR teams use as a watch threshold, so retention deserves a focused review.' if overall > 0.10 else 'This is at or below the 10% level many HR teams use as a watch threshold.'}"
            ),
            "action": "Track this rate monthly by department and review it with HR business partners each quarter.",
        }
    )

    # Highest-attrition department (with at least 50 people, to avoid tiny groups).
    depts = {k: v for k, v in metrics["by_department"].items() if v["n"] >= 50}
    worst_dept = max(depts.items(), key=lambda kv: kv[1]["rate"])
    insights.append(
        {
            "title": f"Department hotspot: {worst_dept[0]}",
            "finding": (
                f"{worst_dept[0]} has the highest attrition at {_pct(worst_dept[1]['rate'])} "
                f"({worst_dept[1]['attrition']} of {worst_dept[1]['n']} employees), "
                f"vs {_pct(overall)} overall."
            ),
            "action": f"Ask HR business partners in {worst_dept[0]} to run stay interviews and review manager-level attrition before the next planning cycle.",
        }
    )

    # Overtime effect.
    ot = metrics["by_overtime"]
    if "Yes" in ot and "No" in ot:
        yes_rate, no_rate = ot["Yes"]["rate"], ot["No"]["rate"]
        ratio = yes_rate / no_rate if no_rate else 0.0
        insights.append(
            {
                "title": "Overtime is the strongest workload signal",
                "finding": (
                    f"Employees working overtime left at {_pct(yes_rate)} "
                    f"({ot['Yes']['attrition']} of {ot['Yes']['n']}), "
                    f"vs {_pct(no_rate)} for employees not working overtime, "
                    f"a {ratio:.1f}x gap."
                ),
                "action": "Cap sustained overtime in the highest-attrition teams, audit overtime rosters monthly, and offer workload relief or schedule flexibility where overtime is chronic.",
            }
        )

    # Tenure band with the highest attrition.
    worst_tenure = max(metrics["by_tenure_band"].items(), key=lambda kv: kv[1]["rate"])
    insights.append(
        {
            "title": f"Early-tenure risk: {worst_tenure[0]} at the company",
            "finding": (
                f"Employees with {worst_tenure[0]} tenure left at {_pct(worst_tenure[1]['rate'])} "
                f"({worst_tenure[1]['attrition']} of {worst_tenure[1]['n']}), the highest of any tenure band."
            ),
            "action": "Strengthen onboarding and the first-year manager check-in cadence, and review pay progression for employees in this tenure band.",
        }
    )

    # Highest-attrition job role (at least 50 people).
    roles = {k: v for k, v in metrics["by_job_role"].items() if v["n"] >= 50}
    worst_role = max(roles.items(), key=lambda kv: kv[1]["rate"])
    insights.append(
        {
            "title": f"Role hotspot: {worst_role[0]}",
            "finding": (
                f"{worst_role[0]} has the highest attrition of any role with 50+ people at "
                f"{_pct(worst_role[1]['rate'])} ({worst_role[1]['attrition']} of {worst_role[1]['n']})."
            ),
            "action": f"Review compensation, workload and career paths for {worst_role[0]} against peer roles before the next compensation cycle.",
        }
    )

    # Leavers vs stayers: tenure and pay.
    lvs = metrics["leavers_vs_stayers"]["leavers"]
    svs = metrics["leavers_vs_stayers"]["stayers"]
    tenure_gap = svs["avg_tenure_years"] - lvs["avg_tenure_years"]
    pay_gap = svs["avg_monthly_income"] - lvs["avg_monthly_income"]
    insights.append(
        {
            "title": "Leavers are shorter-tenured and lower-paid",
            "finding": (
                f"Leavers averaged {lvs['avg_tenure_years']} years at the company vs {svs['avg_tenure_years']} for stayers "
                f"(a {tenure_gap:.1f}-year gap), and ${lvs['avg_monthly_income']:,.0f} average monthly income vs "
                f"${svs['avg_monthly_income']:,.0f} for stayers (a ${pay_gap:,.0f} gap)."
            ),
            "action": "Prioritise retention conversations and pay reviews for lower-paid, shorter-tenured employees in the hotspot roles and departments.",
        }
    )

    # Model drivers.
    drivers = model["top_drivers_random_forest"][:3]
    driver_names = ", ".join(d["feature"] for d in drivers)
    rf_auc = model["random_forest"]["roc_auc"]
    lr_auc = model["logistic_regression"]["roc_auc"]
    best = "random forest" if rf_auc >= lr_auc else "logistic regression"
    insights.append(
        {
            "title": "What the model says drives attrition",
            "finding": (
                f"The top three drivers in the random forest are {driver_names}. "
                f"On the held-out test set the {best} scores ROC-AUC {max(rf_auc, lr_auc):.3f} "
                f"(random forest {rf_auc:.3f}, logistic regression {lr_auc:.3f}), so the model ranks at-risk employees better than chance but is a screening aid, not a decision-maker."
            ),
            "action": "Use the model's ranked risk list to prioritise manager conversations, and re-check the model on new data before acting on any single score.",
        }
    )

    return insights
