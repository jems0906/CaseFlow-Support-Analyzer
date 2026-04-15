from __future__ import annotations

import pandas as pd
import streamlit as st


SCENARIO_PRESETS = {
    "Custom": {
        "description": "Manual scenario controls with no preset assumptions.",
        "resolution_hours_delta": 0,
        "nps_delta": 0.0,
        "adoption_delta": 0.0,
        "renewal_window_days_delta": 0,
    },
    "Support Stabilization": {
        "description": "Moderate service recovery through faster resolution and improved renewal timing.",
        "resolution_hours_delta": -6,
        "nps_delta": 0.4,
        "adoption_delta": 0.03,
        "renewal_window_days_delta": 15,
    },
    "Adoption Lift": {
        "description": "Customer-success intervention focused on activation and workflow usage.",
        "resolution_hours_delta": -2,
        "nps_delta": 0.3,
        "adoption_delta": 0.08,
        "renewal_window_days_delta": 15,
    },
    "NPS Recovery": {
        "description": "Targeted detractor recovery with modest operational improvement.",
        "resolution_hours_delta": -4,
        "nps_delta": 0.8,
        "adoption_delta": 0.04,
        "renewal_window_days_delta": 30,
    },
    "Escalation Risk": {
        "description": "Worsening support conditions, slower renewals, and weaker adoption.",
        "resolution_hours_delta": 8,
        "nps_delta": -0.8,
        "adoption_delta": -0.06,
        "renewal_window_days_delta": -30,
    },
}

SCENARIO_HISTORY_COLUMNS = [
    "scenario_id",
    "scenario_preset",
    "segment_scope",
    "resolution_hours_delta",
    "nps_delta",
    "adoption_delta",
    "renewal_window_days_delta",
    "base_avg_probability",
    "scenario_avg_probability",
    "avg_probability_change",
    "base_high_risk_accounts",
    "scenario_high_risk_accounts",
    "base_arr_at_risk",
    "scenario_arr_at_risk",
    "narrative_summary",
    "account_narrative",
    "driver_narrative",
    "context_narrative",
]


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    export_frame = frame.copy()
    for column in export_frame.columns:
        if pd.api.types.is_datetime64_any_dtype(export_frame[column]):
            export_frame[column] = export_frame[column].dt.strftime("%Y-%m-%d")
    return export_frame.to_csv(index=False).encode("utf-8")


def apply_scenario_preset() -> None:
    preset = SCENARIO_PRESETS[st.session_state["scenario_preset"]]
    st.session_state["resolution_hours_delta"] = preset["resolution_hours_delta"]
    st.session_state["nps_delta"] = preset["nps_delta"]
    st.session_state["adoption_delta"] = preset["adoption_delta"]
    st.session_state["renewal_window_days_delta"] = preset["renewal_window_days_delta"]


def empty_scenario_history_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SCENARIO_HISTORY_COLUMNS)


def initialize_session_state() -> None:
    if "scenario_preset" not in st.session_state:
        st.session_state["scenario_preset"] = "Custom"
        apply_scenario_preset()
    if "scenario_history" not in st.session_state:
        st.session_state["scenario_history"] = []
    if "scenario_history_counter" not in st.session_state:
        st.session_state["scenario_history_counter"] = 1


def filter_frame_by_segments(frame: pd.DataFrame, selected_segments: list[str], *, segment_column: str = "segment") -> pd.DataFrame:
    if not selected_segments:
        return frame.copy()
    return frame[frame[segment_column].isin(selected_segments)]


def current_scenario_history() -> pd.DataFrame:
    if st.session_state["scenario_history"]:
        return pd.DataFrame(st.session_state["scenario_history"])
    return empty_scenario_history_frame()


def build_scenario_summary(
    filtered_scenario_predictions: pd.DataFrame,
    *,
    selected_segments: list[str],
    scenario_preset: str,
    resolution_hours_delta: float,
    nps_delta: float,
    adoption_delta: float,
    renewal_window_days_delta: int,
) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "scenario_preset": scenario_preset,
                "segment_scope": ", ".join(selected_segments) if selected_segments else "All segments",
                "resolution_hours_delta": resolution_hours_delta,
                "nps_delta": nps_delta,
                "adoption_delta": adoption_delta,
                "renewal_window_days_delta": renewal_window_days_delta,
                "base_avg_probability": filtered_scenario_predictions["base_churn_probability"].mean(),
                "scenario_avg_probability": filtered_scenario_predictions["scenario_churn_probability"].mean(),
                "avg_probability_change": filtered_scenario_predictions["probability_change"].mean(),
                "base_high_risk_accounts": int((filtered_scenario_predictions["base_churn_probability"] >= 0.5).sum()),
                "scenario_high_risk_accounts": int((filtered_scenario_predictions["scenario_churn_probability"] >= 0.5).sum()),
                "base_arr_at_risk": filtered_scenario_predictions.loc[filtered_scenario_predictions["base_churn_probability"] >= 0.5, "arr"].sum(),
                "scenario_arr_at_risk": filtered_scenario_predictions.loc[filtered_scenario_predictions["scenario_churn_probability"] >= 0.5, "arr"].sum(),
            }
        ]
    )
    summary.insert(0, "scenario_id", st.session_state["scenario_history_counter"])
    return summary


def build_export_frames(
    *,
    feature_gap: pd.DataFrame,
    filtered_nps: pd.DataFrame,
    filtered_churn: pd.DataFrame,
    customer_directory: pd.DataFrame,
    feature_requests: pd.DataFrame,
    support_theme_overview: pd.DataFrame,
    model_predictions: pd.DataFrame,
    model_feature_importance: pd.DataFrame,
    filtered_scenario_predictions: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    scenario_history: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    export_frames = {
        "Feature gap analysis": feature_gap,
        "NPS trends": filtered_nps,
        "Churn prediction": filtered_churn,
        "Customer directory": customer_directory,
        "Feature request prioritization": feature_requests,
        "Support theme overview": support_theme_overview,
        "Churn model predictions": model_predictions,
        "Churn model feature importance": model_feature_importance,
        "Churn scenario predictions": filtered_scenario_predictions,
        "Churn scenario summary": scenario_summary,
        "Churn scenario history": empty_scenario_history_frame(),
    }
    if not scenario_history.empty:
        export_frames["Churn scenario history"] = scenario_history
    return export_frames