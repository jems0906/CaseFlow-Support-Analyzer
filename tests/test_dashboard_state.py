from types import SimpleNamespace

import pandas as pd

from src.caseflow_support_analyzer import dashboard_state


def test_to_csv_bytes_formats_datetime_columns() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["SMB"],
            "response_month": [pd.Timestamp("2025-01-15")],
        }
    )

    csv_bytes = dashboard_state.to_csv_bytes(frame)

    assert csv_bytes.decode("utf-8").splitlines() == ["segment,response_month", "SMB,2025-01-15"]


def test_filter_frame_by_segments_returns_only_selected_rows() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["SMB", "Enterprise", "SMB"],
            "value": [1, 2, 3],
        }
    )

    filtered = dashboard_state.filter_frame_by_segments(frame, ["SMB"])

    assert filtered["value"].tolist() == [1, 3]


def test_build_scenario_summary_uses_session_counter() -> None:
    dashboard_state.st = SimpleNamespace(session_state={"scenario_history_counter": 7})
    predictions = pd.DataFrame(
        {
            "base_churn_probability": [0.4, 0.7],
            "scenario_churn_probability": [0.3, 0.6],
            "probability_change": [-0.1, -0.1],
            "arr": [1000, 2000],
        }
    )

    summary = dashboard_state.build_scenario_summary(
        predictions,
        selected_segments=["SMB"],
        scenario_preset="Support Stabilization",
        resolution_hours_delta=-6,
        nps_delta=0.4,
        adoption_delta=0.03,
        renewal_window_days_delta=15,
    )

    assert summary.iloc[0]["scenario_id"] == 7
    assert summary.iloc[0]["segment_scope"] == "SMB"
    assert summary.iloc[0]["base_high_risk_accounts"] == 1
    assert summary.iloc[0]["scenario_high_risk_accounts"] == 1
    assert summary.iloc[0]["avg_probability_change"] == -0.1


def test_build_export_frames_uses_empty_history_fallback() -> None:
    empty_frame = pd.DataFrame()

    exports = dashboard_state.build_export_frames(
        feature_gap=empty_frame,
        filtered_nps=empty_frame,
        filtered_churn=empty_frame,
        customer_directory=empty_frame,
        feature_requests=empty_frame,
        support_theme_overview=empty_frame,
        model_predictions=empty_frame,
        model_feature_importance=empty_frame,
        filtered_scenario_predictions=empty_frame,
        scenario_summary=empty_frame,
        scenario_history=empty_frame,
    )

    assert "Churn scenario history" in exports
    assert list(exports["Churn scenario history"].columns) == dashboard_state.SCENARIO_HISTORY_COLUMNS