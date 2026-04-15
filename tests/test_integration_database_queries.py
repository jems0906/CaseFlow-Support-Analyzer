import sqlite3
from pathlib import Path

from src.caseflow_support_analyzer.data_generation import FEATURE_AREAS
from src.caseflow_support_analyzer.database import bootstrap_database
from src.caseflow_support_analyzer.queries import (
    get_churn_prediction,
    get_feature_gap_analysis,
    get_feature_request_prioritization,
    get_kpis,
    get_nps_trends,
)


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}


def test_bootstrap_database_creates_expected_tables_and_view(tmp_path: Path) -> None:
    database_path = tmp_path / "caseflow_support_analyzer.db"

    result_path = bootstrap_database(database_path)

    assert result_path == str(database_path)
    assert database_path.exists()

    with sqlite3.connect(database_path) as connection:
        customer_count = connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        ticket_count = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        nps_count = connection.execute("SELECT COUNT(*) FROM nps_responses").fetchone()[0]
        usage_count = connection.execute("SELECT COUNT(*) FROM usage_logs").fetchone()[0]
        view_count = connection.execute("SELECT COUNT(*) FROM customer_churn_features").fetchone()[0]

        assert customer_count == 250
        assert ticket_count == 5000
        assert nps_count == 3000
        assert usage_count == 18000
        assert view_count == 250

        assert {"ticket_title", "ticket_summary"}.issubset(_table_columns(connection, "tickets"))
        assert {"renewal_risk_signal", "renewal_window_days", "renewal_churned"}.issubset(_table_columns(connection, "customers"))


def test_bootstrap_database_refreshes_outdated_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "stale_caseflow.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, customer_name TEXT)")
        connection.execute("CREATE TABLE tickets (ticket_id INTEGER PRIMARY KEY, customer_id INTEGER, resolution_hours REAL)")
        connection.commit()

    bootstrap_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert {"ticket_title", "ticket_summary"}.issubset(_table_columns(connection, "tickets"))
        assert {"renewal_risk_signal", "renewal_window_days", "renewal_churned"}.issubset(_table_columns(connection, "customers"))
        assert connection.execute("SELECT COUNT(*) FROM customer_churn_features").fetchone()[0] == 250


def test_sql_queries_return_expected_shapes_and_feature_coverage(tmp_path: Path) -> None:
    database_path = Path(bootstrap_database(tmp_path / "query_caseflow.db"))

    kpis = get_kpis(str(database_path))
    feature_gap = get_feature_gap_analysis(str(database_path))
    nps_trends = get_nps_trends(str(database_path))
    churn = get_churn_prediction(str(database_path))
    feature_requests = get_feature_request_prioritization(str(database_path))

    assert kpis.iloc[0]["customer_count"] == 250
    assert kpis.iloc[0]["ticket_count"] == 5000

    assert set(feature_gap["feature_area"]) == set(FEATURE_AREAS)
    assert feature_gap["gap_score"].is_monotonic_decreasing

    assert len(nps_trends) == 48
    assert set(nps_trends["segment"]) == {"SMB", "Mid-Market", "Enterprise", "Strategic"}

    assert len(churn) == 250
    assert churn["churn_risk_score"].is_monotonic_decreasing

    assert set(feature_requests["feature_area"]) == set(FEATURE_AREAS)
    assert feature_requests["priority_score"].is_monotonic_decreasing