from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


QUERY_FILE = Path(__file__).resolve().parents[2] / "sql" / "analysis_queries.sql"


def _load_queries() -> dict[str, str]:
    query_map: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for line in QUERY_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("-- name:"):
            if current_name and current_lines:
                query_map[current_name] = "\n".join(current_lines).strip()
            current_name = line.split(":", 1)[1].strip()
            current_lines = []
            continue
        if current_name:
            current_lines.append(line)

    if current_name and current_lines:
        query_map[current_name] = "\n".join(current_lines).strip()

    return query_map


def _run_query(database_path: str, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(query, connection, params=params)


def get_kpis(database_path: str) -> pd.DataFrame:
    return _run_query(database_path, _load_queries()["kpis"])


def get_feature_gap_analysis(database_path: str) -> pd.DataFrame:
    return _run_query(database_path, _load_queries()["feature_gap"])


def get_nps_trends(database_path: str) -> pd.DataFrame:
    frame = _run_query(database_path, _load_queries()["nps_trends"])
    frame["response_month"] = pd.to_datetime(frame["response_month"])
    return frame


def get_churn_prediction(database_path: str) -> pd.DataFrame:
    return _run_query(database_path, _load_queries()["churn_prediction"])


def get_feature_request_prioritization(database_path: str) -> pd.DataFrame:
    return _run_query(database_path, _load_queries()["feature_requests"])


def get_customer_directory(database_path: str) -> pd.DataFrame:
    query = """
    SELECT
        c.customer_id,
        c.customer_name,
        c.segment,
        c.region,
        c.arr,
        churn.avg_resolution_hours,
        churn.avg_nps,
        churn.adoption_rate,
        churn.churn_risk_score,
        COUNT(t.ticket_id) AS ticket_count,
        SUM(CASE WHEN t.ticket_type = 'Feature Request' THEN 1 ELSE 0 END) AS feature_request_count
    FROM customers c
    JOIN customer_churn_features churn ON churn.customer_id = c.customer_id
    LEFT JOIN tickets t ON t.customer_id = c.customer_id
    GROUP BY
        c.customer_id,
        c.customer_name,
        c.segment,
        c.region,
        c.arr,
        churn.avg_resolution_hours,
        churn.avg_nps,
        churn.adoption_rate,
        churn.churn_risk_score
    ORDER BY churn.churn_risk_score DESC, c.customer_name ASC
    """
    return _run_query(database_path, query)


def get_customer_ticket_breakdown(database_path: str, customer_id: int) -> pd.DataFrame:
    query = """
    SELECT
        feature_area,
        ticket_type,
        severity,
        COUNT(*) AS ticket_count,
        ROUND(AVG(resolution_hours), 2) AS avg_resolution_hours
    FROM tickets
    WHERE customer_id = :customer_id
    GROUP BY feature_area, ticket_type, severity
    ORDER BY ticket_count DESC, avg_resolution_hours DESC
    """
    return _run_query(database_path, query, {"customer_id": customer_id})


def get_customer_nps_history(database_path: str, customer_id: int) -> pd.DataFrame:
    query = """
    SELECT
        response_date,
        score,
        theme,
        comment
    FROM nps_responses
    WHERE customer_id = :customer_id
    ORDER BY response_date ASC
    """
    frame = _run_query(database_path, query, {"customer_id": customer_id})
    frame["response_date"] = pd.to_datetime(frame["response_date"])
    return frame


def get_customer_feature_usage(database_path: str, customer_id: int) -> pd.DataFrame:
    query = """
    SELECT
        feature_area,
        ROUND(AVG(adoption_score), 4) AS adoption_rate,
        AVG(active_users) AS avg_active_users,
        AVG(sessions) AS avg_sessions,
        AVG(actions) AS avg_actions
    FROM usage_logs
    WHERE customer_id = :customer_id
    GROUP BY feature_area
    ORDER BY adoption_rate DESC
    """
    return _run_query(database_path, query, {"customer_id": customer_id})


def get_feature_area_detail(database_path: str, feature_area: str) -> pd.DataFrame:
    query = """
    WITH ticket_agg AS (
        SELECT
            feature_area,
            COUNT(*) AS ticket_count,
            AVG(resolution_hours) AS avg_resolution_hours,
            SUM(CASE WHEN ticket_type = 'Feature Request' THEN 1 ELSE 0 END) AS feature_request_count
        FROM tickets
        WHERE feature_area = :feature_area
        GROUP BY feature_area
    ),
    usage_agg AS (
        SELECT
            feature_area,
            AVG(adoption_score) AS adoption_rate,
            AVG(active_users) AS avg_active_users,
            AVG(sessions) AS avg_sessions
        FROM usage_logs
        WHERE feature_area = :feature_area
        GROUP BY feature_area
    ),
    customer_feature_agg AS (
        SELECT
            t.feature_area,
            t.customer_id,
            MAX(c.arr) AS arr,
            AVG(n.score) AS avg_nps
        FROM tickets t
        JOIN nps_responses n ON n.customer_id = t.customer_id
        JOIN customers c ON c.customer_id = t.customer_id
        WHERE t.feature_area = :feature_area
        GROUP BY t.feature_area, t.customer_id
    ),
    nps_agg AS (
        SELECT
            feature_area,
            AVG(avg_nps) AS avg_nps,
            COUNT(customer_id) AS impacted_customers,
            SUM(arr) AS impacted_arr
        FROM customer_feature_agg
        GROUP BY feature_area
    )
    SELECT
        ticket_agg.feature_area,
        ticket_agg.ticket_count,
        ticket_agg.avg_resolution_hours,
        ticket_agg.feature_request_count,
        usage_agg.adoption_rate,
        usage_agg.avg_active_users,
        usage_agg.avg_sessions,
        nps_agg.avg_nps,
        nps_agg.impacted_customers,
        nps_agg.impacted_arr
    FROM ticket_agg
    JOIN usage_agg ON usage_agg.feature_area = ticket_agg.feature_area
    JOIN nps_agg ON nps_agg.feature_area = ticket_agg.feature_area
    """
    return _run_query(database_path, query, {"feature_area": feature_area})


def get_feature_segment_impact(database_path: str, feature_area: str) -> pd.DataFrame:
    query = """
    WITH ticket_customer_agg AS (
        SELECT
            t.customer_id,
            c.segment,
            COUNT(*) AS ticket_count,
            AVG(t.resolution_hours) AS avg_resolution_hours
        FROM tickets t
        JOIN customers c ON c.customer_id = t.customer_id
        WHERE t.feature_area = :feature_area
        GROUP BY t.customer_id, c.segment
    ),
    nps_customer_agg AS (
        SELECT
            t.customer_id,
            AVG(n.score) AS avg_nps
        FROM tickets t
        JOIN nps_responses n ON n.customer_id = t.customer_id
        WHERE t.feature_area = :feature_area
        GROUP BY t.customer_id
    ),
    usage_customer_agg AS (
        SELECT
            customer_id,
            AVG(adoption_score) AS adoption_rate
        FROM usage_logs
        WHERE feature_area = :feature_area
        GROUP BY customer_id
    )
    SELECT
        ticket_customer_agg.segment,
        SUM(ticket_customer_agg.ticket_count) AS ticket_count,
        COUNT(ticket_customer_agg.customer_id) AS impacted_customers,
        ROUND(AVG(ticket_customer_agg.avg_resolution_hours), 2) AS avg_resolution_hours,
        ROUND(AVG(nps_customer_agg.avg_nps), 2) AS avg_nps,
        ROUND(AVG(usage_customer_agg.adoption_rate), 4) AS adoption_rate
    FROM ticket_customer_agg
    JOIN nps_customer_agg ON nps_customer_agg.customer_id = ticket_customer_agg.customer_id
    JOIN usage_customer_agg ON usage_customer_agg.customer_id = ticket_customer_agg.customer_id
    GROUP BY ticket_customer_agg.segment
    ORDER BY ticket_count DESC
    """
    return _run_query(database_path, query, {"feature_area": feature_area})


def get_feature_monthly_trend(database_path: str, feature_area: str) -> pd.DataFrame:
    query = """
    SELECT
        substr(opened_at, 1, 7) AS ticket_month,
        COUNT(*) AS ticket_count,
        ROUND(AVG(resolution_hours), 2) AS avg_resolution_hours,
        SUM(CASE WHEN ticket_type = 'Feature Request' THEN 1 ELSE 0 END) AS feature_request_count
    FROM tickets
    WHERE feature_area = :feature_area
    GROUP BY substr(opened_at, 1, 7)
    ORDER BY ticket_month ASC
    """
    frame = _run_query(database_path, query, {"feature_area": feature_area})
    frame["ticket_month"] = pd.to_datetime(frame["ticket_month"])
    return frame


def get_support_theme_overview(database_path: str) -> pd.DataFrame:
    query = """
    WITH ticket_theme_map AS (
        SELECT
            ticket_id,
            customer_id,
            resolution_hours,
            CASE
                WHEN lower(ticket_summary) LIKE '%workflow%' OR lower(ticket_summary) LIKE '%routing%' THEN 'Workflow friction'
                WHEN lower(ticket_summary) LIKE '%report%' OR lower(ticket_summary) LIKE '%dashboard%' THEN 'Reporting visibility'
                WHEN lower(ticket_summary) LIKE '%search%' OR lower(ticket_summary) LIKE '%self-service%' OR lower(ticket_summary) LIKE '%article%' THEN 'Self-service experience'
                WHEN lower(ticket_summary) LIKE '%connector%' OR lower(ticket_summary) LIKE '%api%' OR lower(ticket_summary) LIKE '%sync%' THEN 'Integration reliability'
                WHEN lower(ticket_summary) LIKE '%permission%' OR lower(ticket_summary) LIKE '%audit%' OR lower(ticket_summary) LIKE '%workspace%' THEN 'Admin governance'
                ELSE 'General support operations'
            END AS support_theme
        FROM tickets
    ),
    customer_nps AS (
        SELECT customer_id, AVG(score) AS avg_nps
        FROM nps_responses
        GROUP BY customer_id
    ),
    customer_usage AS (
        SELECT customer_id, AVG(adoption_score) AS adoption_rate
        FROM usage_logs
        GROUP BY customer_id
    )
    SELECT
        ticket_theme_map.support_theme,
        COUNT(*) AS ticket_count,
        COUNT(DISTINCT ticket_theme_map.customer_id) AS impacted_customers,
        ROUND(AVG(ticket_theme_map.resolution_hours), 2) AS avg_resolution_hours,
        ROUND(AVG(customer_nps.avg_nps), 2) AS avg_nps,
        ROUND(AVG(customer_usage.adoption_rate), 4) AS adoption_rate
    FROM ticket_theme_map
    JOIN customer_nps ON customer_nps.customer_id = ticket_theme_map.customer_id
    JOIN customer_usage ON customer_usage.customer_id = ticket_theme_map.customer_id
    GROUP BY ticket_theme_map.support_theme
    ORDER BY ticket_count DESC
    """
    return _run_query(database_path, query)


def get_support_nps_theme_alignment(database_path: str) -> pd.DataFrame:
    query = """
    WITH ticket_theme_map AS (
        SELECT DISTINCT
            customer_id,
            CASE
                WHEN lower(ticket_summary) LIKE '%workflow%' OR lower(ticket_summary) LIKE '%routing%' THEN 'Workflow friction'
                WHEN lower(ticket_summary) LIKE '%report%' OR lower(ticket_summary) LIKE '%dashboard%' THEN 'Reporting visibility'
                WHEN lower(ticket_summary) LIKE '%search%' OR lower(ticket_summary) LIKE '%self-service%' OR lower(ticket_summary) LIKE '%article%' THEN 'Self-service experience'
                WHEN lower(ticket_summary) LIKE '%connector%' OR lower(ticket_summary) LIKE '%api%' OR lower(ticket_summary) LIKE '%sync%' THEN 'Integration reliability'
                WHEN lower(ticket_summary) LIKE '%permission%' OR lower(ticket_summary) LIKE '%audit%' OR lower(ticket_summary) LIKE '%workspace%' THEN 'Admin governance'
                ELSE 'General support operations'
            END AS support_theme
        FROM tickets
    )
    SELECT
        ticket_theme_map.support_theme,
        n.theme AS nps_theme,
        COUNT(*) AS linked_feedback_count,
        ROUND(AVG(n.score), 2) AS avg_nps
    FROM ticket_theme_map
    JOIN nps_responses n ON n.customer_id = ticket_theme_map.customer_id
    GROUP BY ticket_theme_map.support_theme, n.theme
    ORDER BY linked_feedback_count DESC, avg_nps ASC
    """
    return _run_query(database_path, query)


def get_support_theme_examples(database_path: str, support_theme: str) -> pd.DataFrame:
    query = """
    WITH ticket_theme_map AS (
        SELECT
            t.ticket_id,
            t.customer_id,
            t.feature_area,
            t.ticket_title,
            t.ticket_summary,
            CASE
                WHEN lower(t.ticket_summary) LIKE '%workflow%' OR lower(t.ticket_summary) LIKE '%routing%' THEN 'Workflow friction'
                WHEN lower(t.ticket_summary) LIKE '%report%' OR lower(t.ticket_summary) LIKE '%dashboard%' THEN 'Reporting visibility'
                WHEN lower(t.ticket_summary) LIKE '%search%' OR lower(t.ticket_summary) LIKE '%self-service%' OR lower(t.ticket_summary) LIKE '%article%' THEN 'Self-service experience'
                WHEN lower(t.ticket_summary) LIKE '%connector%' OR lower(t.ticket_summary) LIKE '%api%' OR lower(t.ticket_summary) LIKE '%sync%' THEN 'Integration reliability'
                WHEN lower(t.ticket_summary) LIKE '%permission%' OR lower(t.ticket_summary) LIKE '%audit%' OR lower(t.ticket_summary) LIKE '%workspace%' THEN 'Admin governance'
                ELSE 'General support operations'
            END AS support_theme
        FROM tickets t
    )
    SELECT
        c.customer_name,
        ticket_theme_map.feature_area,
        ticket_theme_map.ticket_title,
        ticket_theme_map.ticket_summary,
        n.theme AS nps_theme,
        n.comment AS nps_comment
    FROM ticket_theme_map
    JOIN customers c ON c.customer_id = ticket_theme_map.customer_id
    JOIN nps_responses n ON n.customer_id = ticket_theme_map.customer_id
    WHERE ticket_theme_map.support_theme = :support_theme
    ORDER BY n.score ASC, ticket_theme_map.ticket_id ASC
    LIMIT 12
    """
    return _run_query(database_path, query, {"support_theme": support_theme})


def get_churn_model_dataset(database_path: str) -> pd.DataFrame:
    query = """
    WITH ticket_agg AS (
        SELECT
            customer_id,
            COUNT(*) AS ticket_count,
            AVG(resolution_hours) AS avg_resolution_hours,
            SUM(CASE WHEN ticket_type = 'Feature Request' THEN 1 ELSE 0 END) AS feature_request_count,
            SUM(CASE WHEN severity = 'Critical' THEN 1 ELSE 0 END) AS critical_ticket_count
        FROM tickets
        GROUP BY customer_id
    ),
    nps_agg AS (
        SELECT
            customer_id,
            AVG(score) AS avg_nps
        FROM nps_responses
        GROUP BY customer_id
    ),
    usage_agg AS (
        SELECT
            customer_id,
            AVG(adoption_score) AS adoption_rate,
            AVG(active_users) AS avg_active_users,
            AVG(actions) AS avg_actions
        FROM usage_logs
        GROUP BY customer_id
    )
    SELECT
        c.customer_id,
        c.customer_name,
        c.segment,
        c.region,
        c.arr,
        c.seats,
        c.renewal_window_days,
        c.renewal_risk_signal,
        c.renewal_churned,
        ticket_agg.ticket_count,
        ticket_agg.avg_resolution_hours,
        ticket_agg.feature_request_count,
        ticket_agg.critical_ticket_count,
        nps_agg.avg_nps,
        usage_agg.adoption_rate,
        usage_agg.avg_active_users,
        usage_agg.avg_actions
    FROM customers c
    JOIN ticket_agg ON ticket_agg.customer_id = c.customer_id
    JOIN nps_agg ON nps_agg.customer_id = c.customer_id
    JOIN usage_agg ON usage_agg.customer_id = c.customer_id
    ORDER BY c.customer_id
    """
    return _run_query(database_path, query)