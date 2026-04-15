from __future__ import annotations

import sqlite3
from pathlib import Path

from .data_generation import generate_mock_data


def bootstrap_database(database_path: Path) -> str:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_exists = database_path.exists()
    with sqlite3.connect(database_path) as connection:
        needs_refresh = (not database_exists) or _requires_refresh(connection)
        if needs_refresh:
            generated = generate_mock_data()
            generated.customers.to_sql("customers", connection, index=False, if_exists="replace")
            generated.tickets.to_sql("tickets", connection, index=False, if_exists="replace")
            generated.nps_responses.to_sql("nps_responses", connection, index=False, if_exists="replace")
            generated.usage_logs.to_sql("usage_logs", connection, index=False, if_exists="replace")
        _create_views(connection)

    return str(database_path)


def _requires_refresh(connection: sqlite3.Connection) -> bool:
    return any(
        [
            not _table_has_columns(connection, "tickets", {"ticket_title", "ticket_summary"}),
            not _table_has_columns(connection, "customers", {"renewal_risk_signal", "renewal_window_days", "renewal_churned"}),
        ]
    )


def _table_has_columns(connection: sqlite3.Connection, table_name: str, required_columns: set[str]) -> bool:
    cursor = connection.execute(f"PRAGMA table_info({table_name})")
    available_columns = {row[1] for row in cursor.fetchall()}
    return required_columns.issubset(available_columns)


def _create_views(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP VIEW IF EXISTS customer_churn_features;

        CREATE VIEW customer_churn_features AS
        WITH ticket_agg AS (
            SELECT customer_id, AVG(resolution_hours) AS avg_resolution_hours
            FROM tickets
            GROUP BY customer_id
        ),
        nps_agg AS (
            SELECT customer_id, AVG(score) AS avg_nps
            FROM nps_responses
            GROUP BY customer_id
        ),
        usage_agg AS (
            SELECT customer_id, AVG(adoption_score) AS adoption_rate
            FROM usage_logs
            GROUP BY customer_id
        )
        SELECT
            c.customer_id,
            ROUND(ticket_agg.avg_resolution_hours, 2) AS avg_resolution_hours,
            ROUND(nps_agg.avg_nps, 2) AS avg_nps,
            ROUND(usage_agg.adoption_rate, 4) AS adoption_rate,
            ROUND(
                MIN(
                    1.0,
                    MAX(
                        0.0,
                        0.25 * (ticket_agg.avg_resolution_hours / 72.0)
                        + 0.35 * ((10.0 - nps_agg.avg_nps) / 10.0)
                        + 0.40 * (1.0 - usage_agg.adoption_rate)
                    )
                ),
                4
            ) AS churn_risk_score
        FROM customers c
        JOIN ticket_agg ON ticket_agg.customer_id = c.customer_id
        JOIN nps_agg ON nps_agg.customer_id = c.customer_id
        JOIN usage_agg ON usage_agg.customer_id = c.customer_id;
        """
    )
