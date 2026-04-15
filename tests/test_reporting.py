import pandas as pd

from src.caseflow_support_analyzer.reporting import (
    build_board_report_html,
    build_executive_report,
    build_executive_summary,
)


def sample_kpis() -> pd.Series:
    return pd.Series(
        {
            "customer_count": 250,
            "ticket_count": 5000,
            "avg_resolution_hours": 18.4,
            "avg_nps": 7.2,
            "high_risk_customers": 21,
        }
    )


def sample_feature_gap() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_area": "Routing",
                "support_ticket_share": 0.32,
                "adoption_rate": 0.41,
                "gap_score": 1.84,
            }
        ]
    )


def sample_churn() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_name": "Northwind Legal",
                "segment": "SMB",
                "churn_risk_score": 0.74,
                "avg_resolution_hours": 22.3,
                "avg_nps": 5.8,
                "adoption_rate": 0.43,
            },
            {
                "customer_name": "Fabrikam Ops",
                "segment": "Enterprise",
                "churn_risk_score": 0.61,
                "avg_resolution_hours": 19.5,
                "avg_nps": 6.3,
                "adoption_rate": 0.48,
            },
        ]
    )


def sample_feature_requests() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_area": "Routing",
                "priority_score": 2.18,
                "impacted_arr": 420000,
            }
        ]
    )


def test_build_executive_summary_returns_expected_lines() -> None:
    summary = build_executive_summary(sample_kpis(), sample_feature_gap(), sample_churn(), sample_feature_requests())

    assert len(summary) == 4
    assert "5,000 tickets" in summary[0]
    assert "Routing" in summary[1]
    assert "Northwind Legal" in summary[2]


def test_build_executive_report_includes_scenario_sections() -> None:
    scenario_summary = pd.DataFrame(
        [
            {
                "scenario_preset": "Support Stabilization",
                "segment_scope": "SMB",
                "narrative_summary": "Support Stabilization reduced average churn probability by 4.20%.",
                "account_narrative": "Largest account moves: Northwind Legal (SMB) moved down 8.10%.",
                "driver_narrative": "Operational levers: resolution time reduces modeled churn pressure.",
                "context_narrative": "Contextual factors: segment mix: SMB raises modeled churn pressure.",
                "resolution_hours_delta": -6,
                "nps_delta": 0.4,
                "adoption_delta": 0.03,
                "renewal_window_days_delta": 15,
            }
        ]
    )
    scenario_history = pd.DataFrame(
        [
            {
                "scenario_id": 1,
                "scenario_preset": "Support Stabilization",
                "segment_scope": "SMB",
                "narrative_summary": "Saved summary",
                "account_narrative": "Saved account narrative",
                "driver_narrative": "Saved driver narrative",
                "context_narrative": "Saved context narrative",
            }
        ]
    )

    report = build_executive_report(
        sample_kpis(),
        ["Summary line"],
        sample_feature_gap(),
        sample_churn(),
        sample_feature_requests(),
        ["SMB"],
        scenario_summary=scenario_summary,
        scenario_history=scenario_history,
    )

    assert "## Current Scenario Commentary" in report
    assert "## Saved Scenario Snapshots" in report
    assert "Scenario 1: Support Stabilization" in report


def test_build_board_report_html_escapes_user_content() -> None:
    churn = sample_churn().copy()
    churn.loc[0, "customer_name"] = "Northwind <Legal>"

    html_output = build_board_report_html(
        sample_kpis(),
        ["Watch <this> item"],
        sample_feature_gap(),
        churn,
        sample_feature_requests(),
        ["SMB"],
    )

    assert "&lt;Legal&gt;" in html_output
    assert "Watch &lt;this&gt; item" in html_output
    assert "CaseFlow Support Analyzer Board Report" in html_output