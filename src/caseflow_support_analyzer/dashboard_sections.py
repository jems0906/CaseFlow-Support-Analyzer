from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from .dashboard_state import current_scenario_history, to_csv_bytes


def render_report_downloads(
    *,
    summary: list[str],
    export_frames: dict[str, pd.DataFrame],
    executive_report: str,
    board_report_html: str,
) -> None:
    st.subheader("Executive summary")
    for line in summary:
        st.write(f"- {line}")

    export_left, export_right = st.columns((0.75, 0.25))
    with export_left:
        selected_export = st.selectbox("Download dataset", options=list(export_frames.keys()))
    with export_right:
        st.download_button(
            label="Download CSV",
            data=to_csv_bytes(export_frames[selected_export]),
            file_name=f"caseflow_{selected_export.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.download_button(
        label="Download executive report",
        data=executive_report.encode("utf-8"),
        file_name="caseflow_executive_report.md",
        mime="text/markdown",
    )
    st.download_button(
        label="Download board report (HTML)",
        data=board_report_html.encode("utf-8"),
        file_name="caseflow_board_report.html",
        mime="text/html",
    )

    with st.expander("Board report preview", expanded=False):
        st.caption("This HTML layout is print-friendly and can be saved as PDF from the browser print dialog.")
        components.html(board_report_html, height=1100, scrolling=True)


def render_churn_model_section(
    *,
    filtered_model_predictions: pd.DataFrame,
    model_feature_importance: pd.DataFrame,
    model_metrics: pd.Series,
) -> None:
    st.subheader("Churn model")
    model_metric_columns = st.columns(4)
    model_metric_columns[0].metric("Model accuracy", f"{model_metrics['accuracy']:.2f}")
    model_metric_columns[1].metric("ROC AUC", f"{model_metrics['roc_auc']:.2f}")
    model_metric_columns[2].metric("Actual churn rate", f"{model_metrics['baseline_positive_rate']:.2%}")
    model_metric_columns[3].metric("Predicted churn rate", f"{model_metrics['predicted_positive_rate']:.2%}")

    model_left, model_right = st.columns(2)
    with model_left:
        model_scatter = px.scatter(
            filtered_model_predictions,
            x="predicted_churn_probability",
            y="renewal_churned",
            color="segment",
            size="arr",
            hover_data=["customer_name", "region", "prediction_gap"],
            labels={"predicted_churn_probability": "Predicted churn probability", "renewal_churned": "Actual churned"},
        )
        model_scatter.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(model_scatter, use_container_width=True)
    with model_right:
        importance_chart = px.bar(
            model_feature_importance.head(10),
            x="absolute_impact",
            y="feature",
            orientation="h",
            color="coefficient",
            color_continuous_scale="RdBu",
            labels={"absolute_impact": "Absolute coefficient", "feature": "Model feature"},
        )
        importance_chart.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(importance_chart, use_container_width=True)

    st.dataframe(
        filtered_model_predictions.head(15),
        use_container_width=True,
        hide_index=True,
        column_config={
            "arr": st.column_config.NumberColumn(format="$%.0f"),
            "renewal_churned": st.column_config.NumberColumn(format="%d"),
            "predicted_churn_probability": st.column_config.NumberColumn(format="%.2f"),
            "predicted_churn_flag": st.column_config.NumberColumn(format="%d"),
            "prediction_gap": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_scenario_section(
    *,
    filtered_scenario_predictions: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    scenario_history: pd.DataFrame,
) -> pd.DataFrame:
    st.subheader("Scenario simulator")
    st.caption("Use the playbooks for realistic intervention ranges, then fine-tune the controls if you need a custom scenario.")
    scenario_metric_columns = st.columns(4)
    scenario_metric_columns[0].metric(
        "Average churn probability",
        f"{filtered_scenario_predictions['scenario_churn_probability'].mean():.2%}",
        delta=f"{(filtered_scenario_predictions['scenario_churn_probability'].mean() - filtered_scenario_predictions['base_churn_probability'].mean()):.2%}",
    )
    scenario_metric_columns[1].metric(
        "High-risk accounts",
        f"{int((filtered_scenario_predictions['scenario_churn_probability'] >= 0.5).sum()):,}",
        delta=f"{int((filtered_scenario_predictions['scenario_churn_probability'] >= 0.5).sum() - (filtered_scenario_predictions['base_churn_probability'] >= 0.5).sum()):+d}",
    )
    scenario_metric_columns[2].metric(
        "ARR at risk",
        f"${filtered_scenario_predictions.loc[filtered_scenario_predictions['scenario_churn_probability'] >= 0.5, 'arr'].sum():,.0f}",
        delta=f"${filtered_scenario_predictions.loc[filtered_scenario_predictions['scenario_churn_probability'] >= 0.5, 'arr'].sum() - filtered_scenario_predictions.loc[filtered_scenario_predictions['base_churn_probability'] >= 0.5, 'arr'].sum():,.0f}",
    )
    scenario_metric_columns[3].metric(
        "Largest mover",
        filtered_scenario_predictions.iloc[0]["customer_name"],
        delta=f"{filtered_scenario_predictions.iloc[0]['probability_change']:.2%}",
    )

    scenario_summary_row = scenario_summary.iloc[0]
    st.write(
        f"Scenario summary: `{scenario_summary_row['scenario_preset']}` changes average churn probability by {scenario_summary_row['avg_probability_change']:.2%} across `{scenario_summary_row['segment_scope']}`."
    )
    st.write(f"- {scenario_summary_row['narrative_summary']}")
    st.write(f"- {scenario_summary_row['account_narrative']}")
    st.write(f"- {scenario_summary_row['driver_narrative']}")
    st.write(f"- {scenario_summary_row['context_narrative']}")

    scenario_action_left, scenario_action_right = st.columns((0.7, 0.3))
    with scenario_action_left:
        if st.button("Save Scenario Snapshot", use_container_width=True):
            st.session_state["scenario_history"] = [*st.session_state["scenario_history"], scenario_summary.iloc[0].to_dict()][-8:]
            st.session_state["scenario_history_counter"] += 1
    with scenario_action_right:
        if st.button("Clear History", use_container_width=True):
            st.session_state["scenario_history"] = []

    updated_history = current_scenario_history()
    scenario_left, scenario_right = st.columns(2)
    with scenario_left:
        scenario_scatter = px.scatter(
            filtered_scenario_predictions,
            x="base_churn_probability",
            y="scenario_churn_probability",
            color="segment",
            size="arr",
            hover_data=["customer_name", "probability_change"],
            labels={"base_churn_probability": "Baseline probability", "scenario_churn_probability": "Scenario probability"},
        )
        scenario_scatter.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(scenario_scatter, use_container_width=True)
    with scenario_right:
        top_movers = filtered_scenario_predictions.reindex(filtered_scenario_predictions["probability_change"].abs().sort_values(ascending=False).index).head(12)
        scenario_bar = px.bar(
            top_movers,
            x="probability_change",
            y="customer_name",
            orientation="h",
            color="segment",
            labels={"probability_change": "Probability change", "customer_name": "Customer"},
        )
        scenario_bar.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(scenario_bar, use_container_width=True)

    st.dataframe(
        filtered_scenario_predictions.reindex(filtered_scenario_predictions["probability_change"].abs().sort_values(ascending=False).index).head(15),
        use_container_width=True,
        hide_index=True,
        column_config={
            "arr": st.column_config.NumberColumn(format="$%.0f"),
            "renewal_churned": st.column_config.NumberColumn(format="%d"),
            "base_churn_probability": st.column_config.NumberColumn(format="%.2f"),
            "scenario_churn_probability": st.column_config.NumberColumn(format="%.2f"),
            "probability_change": st.column_config.NumberColumn(format="%.2f"),
            "scenario_churn_flag": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.dataframe(
        scenario_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "base_avg_probability": st.column_config.NumberColumn(format="%.2f"),
            "scenario_avg_probability": st.column_config.NumberColumn(format="%.2f"),
            "avg_probability_change": st.column_config.NumberColumn(format="%.2f"),
            "base_arr_at_risk": st.column_config.NumberColumn(format="$%.0f"),
            "scenario_arr_at_risk": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

    st.subheader("Scenario comparison history")
    if updated_history.empty:
        st.caption("Save scenario snapshots to compare interventions side by side.")
    else:
        comparison_left, comparison_right = st.columns(2)
        with comparison_left:
            comparison_frame = pd.concat(
                [
                    updated_history[["scenario_id", "scenario_preset"]].assign(metric="Scenario average probability", value=updated_history["scenario_avg_probability"]),
                    updated_history[["scenario_id", "scenario_preset"]].assign(metric="Baseline average probability", value=updated_history["base_avg_probability"]),
                ],
                ignore_index=True,
            )
            comparison_chart = px.bar(
                comparison_frame,
                x="scenario_id",
                y="value",
                color="metric",
                barmode="group",
                hover_data=["scenario_preset"],
                labels={"scenario_id": "Scenario snapshot", "value": "Probability"},
            )
            comparison_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(comparison_chart, use_container_width=True)
        with comparison_right:
            arr_history_chart = px.bar(
                updated_history,
                x="scenario_id",
                y="scenario_arr_at_risk",
                color="scenario_preset",
                hover_data=["segment_scope", "avg_probability_change"],
                labels={"scenario_id": "Scenario snapshot", "scenario_arr_at_risk": "Scenario ARR at risk"},
            )
            arr_history_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
            st.plotly_chart(arr_history_chart, use_container_width=True)

        st.dataframe(
            updated_history,
            use_container_width=True,
            hide_index=True,
            column_config={
                "base_avg_probability": st.column_config.NumberColumn(format="%.2f"),
                "scenario_avg_probability": st.column_config.NumberColumn(format="%.2f"),
                "avg_probability_change": st.column_config.NumberColumn(format="%.2f"),
                "base_arr_at_risk": st.column_config.NumberColumn(format="$%.0f"),
                "scenario_arr_at_risk": st.column_config.NumberColumn(format="$%.0f"),
            },
        )

        selected_history_id = st.selectbox(
            "Review saved scenario narrative",
            options=updated_history["scenario_id"].tolist(),
            format_func=lambda value: f"Scenario {value}: " + str(updated_history.loc[updated_history["scenario_id"] == value, "scenario_preset"].iloc[0]),
            key="selected_scenario_history_id",
        )
        selected_history_row = updated_history.loc[updated_history["scenario_id"] == selected_history_id].iloc[0]
        st.write(f"- {selected_history_row['narrative_summary']}")
        st.write(f"- {selected_history_row['account_narrative']}")
        st.write(f"- {selected_history_row['driver_narrative']}")
        st.write(f"- {selected_history_row['context_narrative']}")

    return updated_history


def render_analysis_overview(
    *,
    feature_gap: pd.DataFrame,
    feature_requests: pd.DataFrame,
    filtered_nps: pd.DataFrame,
    filtered_churn: pd.DataFrame,
) -> None:
    left_col, right_col = st.columns((1.1, 0.9))

    with left_col:
        st.subheader("Support issues vs. feature gaps")
        gap_chart = px.scatter(
            feature_gap,
            x="adoption_rate",
            y="support_ticket_share",
            size="gap_score",
            color="feature_area",
            hover_data=["avg_resolution_hours", "gap_score"],
            labels={"adoption_rate": "Adoption rate", "support_ticket_share": "Share of support tickets"},
        )
        gap_chart.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(gap_chart, use_container_width=True)
        st.dataframe(
            feature_gap,
            use_container_width=True,
            hide_index=True,
            column_config={
                "adoption_rate": st.column_config.NumberColumn(format="%.2f"),
                "support_ticket_share": st.column_config.NumberColumn(format="%.2f"),
                "avg_resolution_hours": st.column_config.NumberColumn(format="%.1f"),
                "gap_score": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    with right_col:
        st.subheader("Feature request prioritization")
        request_chart = px.bar(
            feature_requests.head(8),
            x="priority_score",
            y="feature_area",
            orientation="h",
            color="impacted_arr",
            color_continuous_scale="Tealgrn",
            labels={"priority_score": "Priority score", "feature_area": "Feature area"},
        )
        request_chart.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(request_chart, use_container_width=True)
        st.dataframe(
            feature_requests,
            use_container_width=True,
            hide_index=True,
            column_config={
                "frequency": st.column_config.NumberColumn(format="%d"),
                "impacted_customers": st.column_config.NumberColumn(format="%d"),
                "impacted_arr": st.column_config.NumberColumn(format="$%.0f"),
                "priority_score": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.subheader("NPS trends by customer segment")
        nps_chart = px.line(
            filtered_nps,
            x="response_month",
            y="avg_nps",
            color="segment",
            markers=True,
            labels={"response_month": "Month", "avg_nps": "Average NPS"},
        )
        nps_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(nps_chart, use_container_width=True)

    with bottom_right:
        st.subheader("Resolution time vs. churn prediction")
        churn_chart = px.scatter(
            filtered_churn,
            x="avg_resolution_hours",
            y="churn_risk_score",
            color="segment",
            size="arr",
            hover_data=["customer_name", "avg_nps", "adoption_rate"],
            labels={"avg_resolution_hours": "Avg resolution hours", "churn_risk_score": "Predicted churn risk"},
        )
        churn_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(churn_chart, use_container_width=True)
        st.dataframe(
            filtered_churn.sort_values("churn_risk_score", ascending=False).head(15),
            use_container_width=True,
            hide_index=True,
            column_config={
                "arr": st.column_config.NumberColumn(format="$%.0f"),
                "avg_resolution_hours": st.column_config.NumberColumn(format="%.1f"),
                "avg_nps": st.column_config.NumberColumn(format="%.1f"),
                "adoption_rate": st.column_config.NumberColumn(format="%.2f"),
                "churn_risk_score": st.column_config.NumberColumn(format="%.2f"),
            },
        )