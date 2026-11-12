# People Analytics Attrition Insights

A personal project that turns HR data into the kind of analysis an Analytics
Business Partner delivers to HR leaders: headcount and turnover metrics,
attrition by segment, an attrition-risk model, and plain-English findings with
recommended actions, all in a Streamlit dashboard.

## What it is

- **HR metrics** (`src/metrics.py`): headcount, overall attrition rate, and
  attrition by department, job role, overtime, tenure band and business travel,
  plus leavers vs stayers on tenure, pay, age and satisfaction.
- **Attrition prediction** (`src/model.py`): logistic regression and random
  forest on a stratified 80/20 train/test split, with ROC-AUC, precision,
  recall and F1 on the held-out test set and the top drivers of attrition.
- **Insights for HR leaders** (`src/insights.py`): a deterministic,
  **rule-based** insights generator. No LLM and no API key are involved; each
  finding is produced by a fixed rule applied to the computed metrics. The
  dashboard labels this as rule-based.
- **Dashboard** (`app.py`): Streamlit app with KPIs, segment charts, model
  metrics and the generated insights.

## Data

IBM HR Analytics Employee Attrition & Performance dataset — 1,470 employees,
35 columns, target `Attrition` (Yes/No). Source: IBM Watson Analytics sample
data, mirrored from Kaggle (`pavansubhasht/ibm-hr-analytics-attrition-dataset`)
via the public GitHub mirror
[`netsatsawat/HR-Analytics`](https://github.com/netsatsawat/HR-Analytics).

**Important:** this is IBM's published *sample* data — fictional employees
created by IBM data scientists for a Watson Analytics demo, not real employees
of any company. It is not Meta data and not real company HR data. The file is
kept in `data/` (224 KB) and is re-fetchable with
`python scripts/download_data.py`, which verifies the SHA-256 before saving.

Verified on download: 1,470 rows, 35 columns, `Attrition` present, 237 Yes /
1,233 No, 0 missing values, SHA-256
`a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659`.

## Verified headline numbers

Measured by running the code (stratified 80/20 split, `random_state=42`;
re-running reproduces these numbers exactly — see `verify_results.py`).

- **Headcount:** 1,470; **attrition:** 237 employees left, **16.1%** overall.
- **Attrition by department:** Sales 20.6% (92 of 446), Human Resources 19.0%
  (12 of 63), Research & Development 13.8% (133 of 961).
- **Attrition by overtime:** overtime Yes 30.5% (127 of 416) vs No 10.4%
  (110 of 1,054) — a 2.9x gap.
- **Attrition by tenure band:** 0–2 yrs 29.8%, 3–5 yrs 13.8%, 6–10 yrs 12.3%,
  11+ yrs 8.1%.
- **Highest-attrition role (50+ people):** Sales Representative 39.8% (33 of 83).
- **Leavers vs stayers:** leavers averaged 5.13 years tenure and $4,787/month
  vs 7.37 years and $6,833/month for stayers.
- **Model (held-out test set, n=294, 47 leavers):**
  - Logistic regression: ROC-AUC **0.808**, precision 0.615, recall 0.340, F1 0.438.
  - Random forest: ROC-AUC **0.799**, precision 0.636, recall 0.149, F1 0.241.
- **Top 3 drivers (random forest):** MonthlyIncome (0.076), Age (0.061),
  TotalWorkingYears (0.058).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_all.py          # metrics + model, writes results/results.json and charts/*.png
python verify_results.py   # re-runs and asserts the numbers match results.json
python -m pytest tests/ -q # 12 tests
streamlit run app.py       # dashboard at http://localhost:8501
```

`results/` regenerates on every run and is git-ignored; the verified numbers
above are recorded here in the README.

## Honest limits

- The dataset is IBM sample data, not real HR data; findings describe that
  sample only and must not be presented as findings about any real company.
- Single cross-sectional snapshot — no time dimension, so no trend or
  forecasting claims.
- No recruiting funnel in the data, so no recruiting metrics are claimed.
- The models are screening aids: test-set recall is low (logistic 0.340,
  random forest 0.149) and neither model should be used to decide anything
  about an individual employee.
- Insights are rule-based templates over computed metrics, not LLM output.
- No R, no Google Sheets macros, no HR project-management experience is
  claimed by this project.
