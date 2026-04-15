from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .analytics import (
    get_customer_feature_usage,
    get_customer_nps_history,
    get_customer_ticket_breakdown,
    get_feature_area_detail,
    get_feature_monthly_trend,
    get_feature_segment_impact,
    get_support_nps_theme_alignment,
    get_support_theme_examples,
    get_support_theme_overview,
)


def render_investigations(
    *,
    database_path: str,
    customer_directory: pd.DataFrame,
    feature_gap: pd.DataFrame,
) -> None:
    st.subheader("Investigations")
    account_tab, feature_tab, theme_tab = st.tabs(["Account drill-down", "Feature drill-down", "Theme synthesis"])

    with account_tab:
        account_options = customer_directory.assign(
            customer_label=lambda frame: frame["customer_name"] + " | " + frame["segment"] + " | risk " + frame["churn_risk_score"].map(lambda value: f"{value:.2f}")
        )
        selected_account_label = st.selectbox("Select account", options=account_options["customer_label"].tolist())
        selected_account = account_options.loc[account_options["customer_label"] == selected_account_label].iloc[0]
        selected_customer_id = int(selected_account["customer_id"])

        account_ticket_breakdown = get_customer_ticket_breakdown(database_path, selected_customer_id)
        account_nps_history = get_customer_nps_history(database_path, selected_customer_id)
        account_feature_usage = get_customer_feature_usage(database_path, selected_customer_id)

        account_metric_columns = st.columns(6)
        account_metric_columns[0].metric("Segment", selected_account["segment"])
        account_metric_columns[1].metric("Region", selected_account["region"])
        account_metric_columns[2].metric("ARR", f"${selected_account['arr']:,.0f}")
        account_metric_columns[3].metric("Tickets", f"{int(selected_account['ticket_count']):,}")
        account_metric_columns[4].metric("Avg NPS", f"{selected_account['avg_nps']:.1f}")
        account_metric_columns[5].metric("Churn risk", f"{selected_account['churn_risk_score']:.2f}")

        account_left, account_right = st.columns(2)
        with account_left:
            usage_chart = px.bar(
                account_feature_usage,
                x="feature_area",
                y="adoption_rate",
                color="avg_actions",
                color_continuous_scale="Blues",
                labels={"feature_area": "Feature area", "adoption_rate": "Adoption rate"},
            )
            usage_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(usage_chart, use_container_width=True)
            st.dataframe(
                account_feature_usage,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "adoption_rate": st.column_config.NumberColumn(format="%.2f"),
                    "avg_active_users": st.column_config.NumberColumn(format="%.1f"),
                    "avg_sessions": st.column_config.NumberColumn(format="%.1f"),
                    "avg_actions": st.column_config.NumberColumn(format="%.1f"),
                },
            )
        with account_right:
            nps_history_chart = px.line(
                account_nps_history,
                x="response_date",
                y="score",
                markers=True,
                labels={"response_date": "Response date", "score": "NPS score"},
            )
            nps_history_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(nps_history_chart, use_container_width=True)
            st.dataframe(account_nps_history, use_container_width=True, hide_index=True)

        st.dataframe(
            account_ticket_breakdown,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticket_count": st.column_config.NumberColumn(format="%d"),
                "avg_resolution_hours": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    with feature_tab:
        selected_feature = st.selectbox("Select feature area", options=feature_gap["feature_area"].tolist())
        feature_detail = get_feature_area_detail(database_path, selected_feature).iloc[0]
        feature_segment_impact = get_feature_segment_impact(database_path, selected_feature)
        feature_monthly_trend = get_feature_monthly_trend(database_path, selected_feature)

        feature_metric_columns = st.columns(6)
        feature_metric_columns[0].metric("Tickets", f"{int(feature_detail['ticket_count']):,}")
        feature_metric_columns[1].metric("Feature requests", f"{int(feature_detail['feature_request_count']):,}")
        feature_metric_columns[2].metric("Avg resolution", f"{feature_detail['avg_resolution_hours']:.1f} hrs")
        feature_metric_columns[3].metric("Adoption", f"{feature_detail['adoption_rate']:.2f}")
        feature_metric_columns[4].metric("Avg NPS", f"{feature_detail['avg_nps']:.1f}")
        feature_metric_columns[5].metric("Impacted ARR", f"${feature_detail['impacted_arr']:,.0f}")

        feature_left, feature_right = st.columns(2)
        with feature_left:
            monthly_chart = px.bar(
                feature_monthly_trend,
                x="ticket_month",
                y="ticket_count",
                color="feature_request_count",
                color_continuous_scale="Tealgrn",
                labels={"ticket_month": "Month", "ticket_count": "Ticket count"},
            )
            monthly_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(monthly_chart, use_container_width=True)
            st.dataframe(
                feature_monthly_trend,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "avg_resolution_hours": st.column_config.NumberColumn(format="%.1f"),
                    "feature_request_count": st.column_config.NumberColumn(format="%d"),
                },
            )
        with feature_right:
            segment_chart = px.scatter(
                feature_segment_impact,
                x="adoption_rate",
                y="ticket_count",
                size="impacted_customers",
                color="segment",
                hover_data=["avg_nps", "avg_resolution_hours"],
                labels={"adoption_rate": "Adoption rate", "ticket_count": "Ticket count"},
            )
            segment_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(segment_chart, use_container_width=True)
            st.dataframe(
                feature_segment_impact,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "avg_resolution_hours": st.column_config.NumberColumn(format="%.1f"),
                    "avg_nps": st.column_config.NumberColumn(format="%.1f"),
                    "adoption_rate": st.column_config.NumberColumn(format="%.2f"),
                },
            )

    with theme_tab:
        support_theme_overview = get_support_theme_overview(database_path)
        support_theme_alignment = get_support_nps_theme_alignment(database_path)
        selected_support_theme = st.selectbox("Select support theme", options=support_theme_overview["support_theme"].tolist())
        theme_examples = get_support_theme_examples(database_path, selected_support_theme)
        filtered_alignment = support_theme_alignment[support_theme_alignment["support_theme"] == selected_support_theme]

        theme_metrics = support_theme_overview.loc[support_theme_overview["support_theme"] == selected_support_theme].iloc[0]
        theme_metric_columns = st.columns(5)
        theme_metric_columns[0].metric("Tickets", f"{int(theme_metrics['ticket_count']):,}")
        theme_metric_columns[1].metric("Impacted customers", f"{int(theme_metrics['impacted_customers']):,}")
        theme_metric_columns[2].metric("Avg resolution", f"{theme_metrics['avg_resolution_hours']:.1f} hrs")
        theme_metric_columns[3].metric("Avg NPS", f"{theme_metrics['avg_nps']:.1f}")
        theme_metric_columns[4].metric("Adoption", f"{theme_metrics['adoption_rate']:.2f}")

        theme_left, theme_right = st.columns(2)
        with theme_left:
            overview_chart = px.bar(
                support_theme_overview,
                x="ticket_count",
                y="support_theme",
                orientation="h",
                color="avg_nps",
                color_continuous_scale="RdYlGn",
                labels={"ticket_count": "Ticket count", "support_theme": "Support theme"},
            )
            overview_chart.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(overview_chart, use_container_width=True)
            st.dataframe(
                support_theme_overview,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "avg_resolution_hours": st.column_config.NumberColumn(format="%.1f"),
                    "avg_nps": st.column_config.NumberColumn(format="%.1f"),
                    "adoption_rate": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        with theme_right:
            alignment_chart = px.bar(
                filtered_alignment,
                x="linked_feedback_count",
                y="nps_theme",
                orientation="h",
                color="avg_nps",
                color_continuous_scale="Tealgrn",
                labels={"linked_feedback_count": "Linked feedback", "nps_theme": "NPS theme"},
            )
            alignment_chart.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(alignment_chart, use_container_width=True)
            st.dataframe(
                filtered_alignment,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "linked_feedback_count": st.column_config.NumberColumn(format="%d"),
                    "avg_nps": st.column_config.NumberColumn(format="%.1f"),
                },
            )

        st.dataframe(theme_examples, use_container_width=True, hide_index=True)