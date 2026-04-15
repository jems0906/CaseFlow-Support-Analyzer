# CaseFlow Support Analyzer

CaseFlow Support Analyzer is a Streamlit analytics app for combining support tickets, NPS feedback, and product usage data into a single decision-support dashboard.

## What it includes

- Feature-gap analysis that correlates support issue volume with low product adoption
- NPS trend tracking by customer segment
- Churn risk analysis using ticket resolution time, NPS, and usage signals
- Feature request prioritization based on frequency and business impact
- Executive summary generation for leadership reviews
- CSV exports for dashboard datasets and a downloadable executive report
- Account and feature drill-down views for deeper investigation
- Support theme synthesis across ticket text and NPS feedback comments
- Trained churn model with feature importance and prediction review
- Churn scenario simulator for testing resolution, NPS, adoption, and renewal-window changes
- Board-ready HTML report preview and download path for print/PDF workflows
- A reproducible mock dataset with 5,000 support tickets, NPS responses, and usage logs

## Project structure

```text
.
|-- app.py
|-- requirements.txt
|-- tests/
|-- sql/
|   `-- analysis_queries.sql
`-- src/
    `-- caseflow_support_analyzer/
        |-- __init__.py
        |-- analytics.py
    |-- dashboard.py
    |-- dashboard_investigations.py
    |-- dashboard_sections.py
    |-- dashboard_state.py
        |-- data_generation.py
    |-- database.py
    |-- modeling.py
    |-- queries.py
    `-- reporting.py
```

## Run locally

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the dashboard:

```bash
streamlit run app.py
```

4. Run the automated tests:

```bash
pytest
```

The app generates a local SQLite database and mock data on first run.

Use the controls beneath the executive summary to download filtered dashboard data as CSV or generate a Markdown executive report for leadership sharing.
The app also provides a board-ready HTML report that can be printed to PDF from the browser.

## Data model

- `customers`: customer profile, segment, ARR, seats, lifecycle dates
- `tickets`: support tickets with feature area, severity, and resolution time
- `nps_responses`: score history and feedback themes
- `usage_logs`: monthly feature usage and adoption signals

## Notes

- The churn score is a transparent heuristic derived from resolution times, detractor rate, and feature adoption.
- The churn model is a logistic regression trained on synthetic renewal outcomes derived from support, NPS, usage, and renewal-window signals.
- The scenario simulator includes preset intervention playbooks to keep what-if analysis within more realistic operating ranges.
- Saved scenarios include business-readable narratives based on largest account movers and top churn drivers.
- Exported executive reports include current scenario commentary and saved scenario snapshot narratives.
- Scenario narratives distinguish operational levers from static contextual factors.
- Core dashboard datasets are produced with SQL queries against SQLite, then visualized in Streamlit.
- Lightweight pytest coverage is included for reporting, modeling, and dashboard helper functions.
