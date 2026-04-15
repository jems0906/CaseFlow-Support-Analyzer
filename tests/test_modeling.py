import pandas as pd

from src.caseflow_support_analyzer import modeling


def sample_model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": 1,
                "customer_name": "Northwind Legal",
                "segment": "SMB",
                "region": "NA",
                "arr": 12000,
                "seats": 45,
                "renewal_window_days": 90,
                "renewal_risk_signal": 0.62,
                "renewal_churned": 1,
                "ticket_count": 22,
                "avg_resolution_hours": 28.0,
                "feature_request_count": 7,
                "critical_ticket_count": 3,
                "avg_nps": 4.5,
                "adoption_rate": 0.32,
                "avg_active_users": 18.0,
                "avg_actions": 110.0,
            },
            {
                "customer_id": 2,
                "customer_name": "Fabrikam Ops",
                "segment": "Enterprise",
                "region": "EMEA",
                "arr": 54000,
                "seats": 180,
                "renewal_window_days": 120,
                "renewal_risk_signal": 0.24,
                "renewal_churned": 0,
                "ticket_count": 8,
                "avg_resolution_hours": 10.0,
                "feature_request_count": 2,
                "critical_ticket_count": 0,
                "avg_nps": 8.3,
                "adoption_rate": 0.76,
                "avg_active_users": 95.0,
                "avg_actions": 420.0,
            },
            {
                "customer_id": 3,
                "customer_name": "Contoso Services",
                "segment": "Mid-Market",
                "region": "NA",
                "arr": 26000,
                "seats": 90,
                "renewal_window_days": 75,
                "renewal_risk_signal": 0.55,
                "renewal_churned": 1,
                "ticket_count": 18,
                "avg_resolution_hours": 24.0,
                "feature_request_count": 5,
                "critical_ticket_count": 2,
                "avg_nps": 5.1,
                "adoption_rate": 0.39,
                "avg_active_users": 36.0,
                "avg_actions": 160.0,
            },
            {
                "customer_id": 4,
                "customer_name": "Adventure Matters",
                "segment": "Strategic",
                "region": "APAC",
                "arr": 87000,
                "seats": 260,
                "renewal_window_days": 150,
                "renewal_risk_signal": 0.18,
                "renewal_churned": 0,
                "ticket_count": 6,
                "avg_resolution_hours": 8.0,
                "feature_request_count": 1,
                "critical_ticket_count": 0,
                "avg_nps": 9.0,
                "adoption_rate": 0.84,
                "avg_active_users": 130.0,
                "avg_actions": 510.0,
            },
        ]
    )


def test_prepare_churn_model_features_reuses_training_schema() -> None:
    frame = sample_model_frame()
    design_matrix, feature_names, numeric_means, numeric_stds = modeling._prepare_churn_model_features(frame)
    transformed_again, repeated_names, _, _ = modeling._prepare_churn_model_features(
        frame.iloc[[0, 1]],
        feature_names=feature_names,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
    )

    assert list(design_matrix.columns) == feature_names
    assert list(transformed_again.columns) == repeated_names
    assert transformed_again.shape[1] == design_matrix.shape[1]


def test_train_churn_model_returns_predictions_and_metrics(monkeypatch) -> None:
    monkeypatch.setattr(modeling, "get_churn_model_dataset", lambda _: pd.concat([sample_model_frame()] * 8, ignore_index=True))

    result = modeling.train_churn_model("unused.db")

    assert len(result.predictions) == 32
    assert set(result.metrics) == {"accuracy", "roc_auc", "baseline_positive_rate", "predicted_positive_rate"}
    assert result.feature_importance.iloc[0]["absolute_impact"] >= result.feature_importance.iloc[-1]["absolute_impact"]


def test_build_scenario_narrative_returns_all_sections() -> None:
    scenario_summary_row = {
        "scenario_preset": "Support Stabilization",
        "avg_probability_change": -0.042,
        "scenario_arr_at_risk": 180000.0,
        "base_arr_at_risk": 230000.0,
        "scenario_high_risk_accounts": 12,
        "base_high_risk_accounts": 15,
    }
    scenario_predictions = pd.DataFrame(
        [
            {"customer_name": "Northwind Legal", "segment": "SMB", "probability_change": -0.081},
            {"customer_name": "Contoso Services", "segment": "Mid-Market", "probability_change": -0.063},
            {"customer_name": "Fabrikam Ops", "segment": "Enterprise", "probability_change": -0.018},
        ]
    )
    feature_importance = pd.DataFrame(
        [
            {"feature": "avg_resolution_hours", "coefficient": 0.9},
            {"feature": "avg_nps", "coefficient": -0.8},
            {"feature": "adoption_rate", "coefficient": -0.7},
            {"feature": "segment_SMB", "coefficient": 0.3},
        ]
    )

    narrative = modeling.build_scenario_narrative(scenario_summary_row, scenario_predictions, feature_importance)

    assert set(narrative) == {"narrative_summary", "account_narrative", "driver_narrative", "context_narrative"}
    assert "Support Stabilization reduced average churn probability" in narrative["narrative_summary"]
    assert "Northwind Legal" in narrative["account_narrative"]
    assert "Operational levers" in narrative["driver_narrative"]