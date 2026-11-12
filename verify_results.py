"""Re-run metrics + model and assert the headline numbers match results.json.

Run after `python run_all.py`. Exits non-zero on any mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import load_data
from src.metrics import compute_metrics
from src.model import train_and_evaluate

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "results.json"


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    stored = json.loads(RESULTS_PATH.read_text())
    df = load_data()
    metrics = compute_metrics(df)
    model = train_and_evaluate(df)

    checks = []
    checks.append(("headcount", metrics["headcount"] == stored["metrics"]["headcount"]))
    checks.append(("attrition_count", metrics["attrition_count"] == stored["metrics"]["attrition_count"]))
    checks.append(("attrition_rate", _close(metrics["attrition_rate"], stored["metrics"]["attrition_rate"])))
    for dept, seg in metrics["by_department"].items():
        checks.append((f"dept {dept} rate", _close(seg["rate"], stored["metrics"]["by_department"][dept]["rate"])))
    for key in ("Yes", "No"):
        checks.append(
            (f"overtime {key} rate", _close(metrics["by_overtime"][key]["rate"], stored["metrics"]["by_overtime"][key]["rate"]))
        )
    for name in ("logistic_regression", "random_forest"):
        for metric in ("roc_auc", "precision", "recall", "f1"):
            checks.append(
                (
                    f"{name} {metric}",
                    _close(model[name][metric], stored["model"][name][metric]),
                )
            )
    top_stored = [d["feature"] for d in stored["model"]["top_drivers_random_forest"][:3]]
    top_now = [d["feature"] for d in model["top_drivers_random_forest"][:3]]
    checks.append(("top 3 drivers", top_stored == top_now))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        print(f"\n{len(failed)} check(s) failed: {failed}")
        raise SystemExit(1)
    print(f"\nALL CHECKS PASSED ({len(checks)}/{len(checks)})")


if __name__ == "__main__":
    main()
