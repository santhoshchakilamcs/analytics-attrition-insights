"""Metrics correctness on the real IBM HR Analytics data."""
from src.insights import generate_insights


def test_headcount_is_1470(df):
    assert len(df) == 1470


def test_attrition_rate_matches_counts(metrics):
    assert metrics["headcount"] == 1470
    assert metrics["attrition_count"] == 237
    assert metrics["stayed_count"] == 1233
    assert metrics["attrition_rate"] == round(237 / 1470, 4)


def test_department_segments_sum_to_headcount(metrics):
    total_n = sum(v["n"] for v in metrics["by_department"].values())
    total_left = sum(v["attrition"] for v in metrics["by_department"].values())
    assert total_n == 1470
    assert total_left == 237
    assert set(metrics["by_department"]) == {"Sales", "Research & Development", "Human Resources"}


def test_overtime_rates_in_valid_range(metrics):
    for segment in metrics["by_overtime"].values():
        assert 0.0 <= segment["rate"] <= 1.0
        assert segment["n"] > 0
    assert metrics["by_overtime"]["Yes"]["rate"] > metrics["by_overtime"]["No"]["rate"]


def test_tenure_bands_cover_all_employees(metrics):
    total = sum(v["n"] for v in metrics["by_tenure_band"].values())
    assert total == 1470
    assert set(metrics["by_tenure_band"]) == {"0-2 yrs", "3-5 yrs", "6-10 yrs", "11+ yrs"}


def test_leavers_vs_stayers_are_numeric_and_positive(metrics):
    for group in ("leavers", "stayers"):
        g = metrics["leavers_vs_stayers"][group]
        assert g["n"] > 0
        assert g["avg_tenure_years"] > 0
        assert g["avg_monthly_income"] > 0
        assert 18 <= g["avg_age"] <= 65


def test_insights_cover_real_segments(metrics, model):
    insights = generate_insights(metrics, model)
    assert len(insights) >= 6
    titles = " ".join(i["title"] for i in insights)
    assert "Overtime" in titles
    assert "Department hotspot" in titles
    for item in insights:
        assert item["title"] and item["finding"] and item["action"]
    # The overall-turnover finding states the real headcount and leaver count.
    assert "237" in insights[0]["finding"] and "1470" in insights[0]["finding"]
