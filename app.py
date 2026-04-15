from pathlib import Path

import pandas as pd
import streamlit as st

from src.caseflow_support_analyzer.analytics import (
    build_board_report_html,
    build_executive_report,
    build_executive_summary,
    build_scenario_narrative,
    get_customer_directory,
    get_churn_prediction,
    get_feature_gap_analysis,
    get_feature_request_prioritization,
    get_kpis,
    get_nps_trends,
    get_support_theme_overview,
    simulate_churn_scenario,
    train_churn_model,
)
from src.caseflow_support_analyzer.dashboard import (
    SCENARIO_PRESETS,
    apply_scenario_preset,
    build_export_frames,
    build_scenario_summary,
    current_scenario_history,
    filter_frame_by_segments,
    initialize_session_state,
    render_analysis_overview,
    render_churn_model_section,
    render_investigations,
    render_report_downloads,
    render_scenario_section,
)
from src.caseflow_support_analyzer.database import bootstrap_database


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "data" / "caseflow_support_analyzer.db"


st.set_page_config(
    page_title="CaseFlow Support Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()


@st.cache_resource(show_spinner=False)
def load_database() -> str:
    return bootstrap_database(DATABASE_PATH)


@st.cache_data(show_spinner=False)
def load_datasets(database_path: str) -> dict[str, pd.DataFrame]:
    return {
        "kpis": get_kpis(database_path),
        "feature_gap": get_feature_gap_analysis(database_path),
        "nps_trends": get_nps_trends(database_path),
        "churn": get_churn_prediction(database_path),
        "customers": get_customer_directory(database_path),
        "feature_requests": get_feature_request_prioritization(database_path),
        "support_theme_overview": get_support_theme_overview(database_path),
    }


@st.cache_data(show_spinner=False)
def load_churn_model(database_path: str):
    return train_churn_model(database_path)


database_path = load_database()
datasets = load_datasets(database_path)
churn_model = load_churn_model(database_path)

kpis = datasets["kpis"].iloc[0]
feature_gap = datasets["feature_gap"]
nps_trends = datasets["nps_trends"]
churn = datasets["churn"]
customer_directory = datasets["customers"]
feature_requests = datasets["feature_requests"]
support_theme_overview = datasets["support_theme_overview"]
model_predictions = churn_model.predictions
model_feature_importance = churn_model.feature_importance
model_metrics = pd.Series(churn_model.metrics)

st.title("CaseFlow Support Analyzer")
st.caption(
    "Support intelligence across tickets, NPS, and usage logs for product and customer success teams."
)

with st.sidebar:
    st.header("Filters")
    segments = sorted(churn["segment"].unique().tolist())
    selected_segments = st.multiselect("Customer segments", options=segments, default=segments)
    st.header("Scenario preset")
    st.selectbox(
        "Intervention playbook",
        options=list(SCENARIO_PRESETS.keys()),
        key="scenario_preset",
        on_change=apply_scenario_preset,
    )
    st.caption(SCENARIO_PRESETS[st.session_state["scenario_preset"]]["description"])

filtered_churn = filter_frame_by_segments(churn, selected_segments)
filtered_nps = filter_frame_by_segments(nps_trends, selected_segments)
filtered_model_predictions = filter_frame_by_segments(model_predictions, selected_segments)

scenario_control_columns = st.columns(4)
resolution_hours_delta = scenario_control_columns[0].slider(
    "Resolution time change (hrs)",
    min_value=-24,
    max_value=24,
    step=2,
    key="resolution_hours_delta",
)
nps_delta = scenario_control_columns[1].slider(
    "NPS change",
    min_value=-2.0,
    max_value=2.0,
    step=0.2,
    key="nps_delta",
)
adoption_delta = scenario_control_columns[2].slider(
    "Adoption change",
    min_value=-0.2,
    max_value=0.2,
    step=0.02,
    key="adoption_delta",
)
renewal_window_days_delta = scenario_control_columns[3].slider(
    "Renewal window change (days)",
    min_value=-90,
    max_value=90,
    step=15,
    key="renewal_window_days_delta",
)

scenario_predictions = simulate_churn_scenario(
    database_path,
    churn_model,
    resolution_hours_delta=resolution_hours_delta,
    nps_delta=nps_delta,
    adoption_delta=adoption_delta,
    renewal_window_days_delta=renewal_window_days_delta,
)
filtered_scenario_predictions = filter_frame_by_segments(scenario_predictions, selected_segments)

scenario_summary = build_scenario_summary(
    filtered_scenario_predictions,
    selected_segments=selected_segments,
    scenario_preset=st.session_state["scenario_preset"],
    resolution_hours_delta=resolution_hours_delta,
    nps_delta=nps_delta,
    adoption_delta=adoption_delta,
    renewal_window_days_delta=renewal_window_days_delta,
)
scenario_narrative = build_scenario_narrative(
    scenario_summary.iloc[0],
    filtered_scenario_predictions,
    model_feature_importance,
)
for column_name, value in scenario_narrative.items():
    scenario_summary[column_name] = value

scenario_history = current_scenario_history()

metric_columns = st.columns(5)
metric_columns[0].metric("Customers", f"{int(kpis['customer_count']):,}")
metric_columns[1].metric("Tickets", f"{int(kpis['ticket_count']):,}")
metric_columns[2].metric("Avg resolution", f"{kpis['avg_resolution_hours']:.1f} hrs")
metric_columns[3].metric("Average NPS", f"{kpis['avg_nps']:.1f}")
metric_columns[4].metric("High churn risk", f"{int(kpis['high_risk_customers']):,}")

summary = build_executive_summary(
    kpis=kpis,
    feature_gap=feature_gap,
    churn=filtered_churn,
    feature_requests=feature_requests,
)
executive_report = build_executive_report(
    kpis=kpis,
    summary=summary,
    feature_gap=feature_gap,
    churn=filtered_churn,
    feature_requests=feature_requests,
    selected_segments=selected_segments,
    scenario_summary=scenario_summary,
    scenario_history=scenario_history,
)
board_report_html = build_board_report_html(
    kpis=kpis,
    summary=summary,
    feature_gap=feature_gap,
    churn=filtered_churn,
    feature_requests=feature_requests,
    selected_segments=selected_segments,
    scenario_summary=scenario_summary,
    scenario_history=scenario_history,
)

export_frames = build_export_frames(
    feature_gap=feature_gap,
    filtered_nps=filtered_nps,
    filtered_churn=filtered_churn,
    customer_directory=customer_directory,
    feature_requests=feature_requests,
    support_theme_overview=support_theme_overview,
    model_predictions=model_predictions,
    model_feature_importance=model_feature_importance,
    filtered_scenario_predictions=filtered_scenario_predictions,
    scenario_summary=scenario_summary,
    scenario_history=scenario_history,
)

render_report_downloads(
    summary=summary,
    export_frames=export_frames,
    executive_report=executive_report,
    board_report_html=board_report_html,
)
render_churn_model_section(
    filtered_model_predictions=filtered_model_predictions,
    model_feature_importance=model_feature_importance,
    model_metrics=model_metrics,
)
render_scenario_section(
    filtered_scenario_predictions=filtered_scenario_predictions,
    scenario_summary=scenario_summary,
    scenario_history=scenario_history,
)
render_analysis_overview(
    feature_gap=feature_gap,
    feature_requests=feature_requests,
    filtered_nps=filtered_nps,
    filtered_churn=filtered_churn,
)
render_investigations(
    database_path=database_path,
    customer_directory=customer_directory,
    feature_gap=feature_gap,
)