from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .queries import get_churn_model_dataset


CHURN_NUMERIC_FEATURES = [
    "arr",
    "seats",
    "renewal_window_days",
    "renewal_risk_signal",
    "ticket_count",
    "avg_resolution_hours",
    "feature_request_count",
    "critical_ticket_count",
    "avg_nps",
    "adoption_rate",
    "avg_active_users",
    "avg_actions",
]
CHURN_CATEGORICAL_FEATURES = ["segment", "region"]


@dataclass(frozen=True)
class ChurnModelResult:
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    metrics: dict[str, float]
    weights: np.ndarray
    bias: float
    feature_names: list[str]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def _accuracy_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float((actual == predicted).mean())


def _roc_auc_score(actual: np.ndarray, probabilities: np.ndarray) -> float:
    positive_mask = actual == 1
    negative_mask = actual == 0
    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())
    if positive_count == 0 or negative_count == 0:
        return 0.5
    order = np.argsort(probabilities)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probabilities) + 1)
    positive_rank_sum = ranks[positive_mask].sum()
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(auc)


def _prepare_churn_model_features(
    frame: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    numeric_means: dict[str, float] | None = None,
    numeric_stds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, float], dict[str, float]]:
    model_frame = frame[CHURN_NUMERIC_FEATURES + CHURN_CATEGORICAL_FEATURES].copy()
    model_frame[CHURN_NUMERIC_FEATURES] = model_frame[CHURN_NUMERIC_FEATURES].fillna(model_frame[CHURN_NUMERIC_FEATURES].median())
    model_frame[CHURN_CATEGORICAL_FEATURES] = model_frame[CHURN_CATEGORICAL_FEATURES].fillna("Unknown")

    if numeric_means is None:
        numeric_mean_series = model_frame[CHURN_NUMERIC_FEATURES].mean()
    else:
        numeric_mean_series = pd.Series(numeric_means, index=CHURN_NUMERIC_FEATURES, dtype=float)

    if numeric_stds is None:
        numeric_std_series = model_frame[CHURN_NUMERIC_FEATURES].std(ddof=0).replace(0, 1)
    else:
        numeric_std_series = pd.Series(numeric_stds, index=CHURN_NUMERIC_FEATURES, dtype=float).replace(0, 1)

    scaled_numeric = (model_frame[CHURN_NUMERIC_FEATURES] - numeric_mean_series) / numeric_std_series
    encoded_categorical = pd.get_dummies(model_frame[CHURN_CATEGORICAL_FEATURES], prefix=CHURN_CATEGORICAL_FEATURES, dtype=float)
    design_matrix = pd.concat([scaled_numeric, encoded_categorical], axis=1)
    resolved_feature_names = feature_names or list(design_matrix.columns)
    design_matrix = design_matrix.reindex(columns=resolved_feature_names, fill_value=0.0)
    return design_matrix, resolved_feature_names, numeric_mean_series.to_dict(), numeric_std_series.to_dict()


def _fit_logistic_regression(features: np.ndarray, target: np.ndarray, *, iterations: int = 4000, learning_rate: float = 0.08) -> tuple[np.ndarray, float]:
    sample_count, feature_count = features.shape
    weights = np.zeros(feature_count, dtype=float)
    bias = 0.0
    positive_weight = sample_count / max(target.sum() * 2.0, 1.0)
    negative_weight = sample_count / max((sample_count - target.sum()) * 2.0, 1.0)
    sample_weights = np.where(target == 1, positive_weight, negative_weight)

    for _ in range(iterations):
        predictions = _sigmoid(features @ weights + bias)
        errors = (predictions - target) * sample_weights
        gradient_weights = (features.T @ errors) / sample_count
        gradient_bias = float(errors.mean())
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    return weights, bias


def _compute_renewal_risk_signal(frame: pd.DataFrame) -> pd.Series:
    segment_risk_bias = frame["segment"].map({"SMB": 0.05, "Mid-Market": 0.02, "Enterprise": 0.04, "Strategic": 0.06}).fillna(0.03)
    signal = (
        0.22 * np.minimum(frame["avg_resolution_hours"] / 72.0, 1.5)
        + 0.32 * ((10.0 - frame["avg_nps"]) / 10.0)
        + 0.26 * (1.0 - frame["adoption_rate"])
        + 0.12 * np.minimum(frame["feature_request_count"] / 12.0, 1.2)
        + 0.08 * np.minimum(frame["critical_ticket_count"] / 5.0, 1.0)
        + segment_risk_bias
    )
    return signal.clip(0.02, 0.98)


def train_churn_model(database_path: str) -> ChurnModelResult:
    frame = get_churn_model_dataset(database_path)
    target = frame["renewal_churned"].astype(int).to_numpy()
    design_matrix, feature_names, numeric_means, numeric_stds = _prepare_churn_model_features(frame)

    rng = np.random.default_rng(42)
    positive_indices = np.flatnonzero(target == 1)
    negative_indices = np.flatnonzero(target == 0)
    positive_test_size = max(1, min(len(positive_indices), int(round(len(positive_indices) * 0.25))))
    negative_test_size = max(1, min(len(negative_indices), int(round(len(negative_indices) * 0.25))))
    test_indices = np.concatenate(
        [
            rng.choice(positive_indices, size=positive_test_size, replace=False),
            rng.choice(negative_indices, size=negative_test_size, replace=False),
        ]
    )
    train_mask = np.ones(len(frame), dtype=bool)
    train_mask[test_indices] = False
    test_mask = ~train_mask

    x_train = design_matrix.loc[train_mask, feature_names].to_numpy(dtype=float)
    x_test = design_matrix.loc[test_mask, feature_names].to_numpy(dtype=float)
    y_train = target[train_mask]
    y_test = target[test_mask]

    weights, bias = _fit_logistic_regression(x_train, y_train)
    test_probabilities = _sigmoid(x_test @ weights + bias)
    test_predictions = (test_probabilities >= 0.5).astype(int)
    full_probabilities = _sigmoid(design_matrix.loc[:, feature_names].to_numpy(dtype=float) @ weights + bias)

    full_predictions = frame[["customer_id", "customer_name", "segment", "region", "arr", "renewal_churned"]].copy()
    full_predictions["predicted_churn_probability"] = full_probabilities
    full_predictions["predicted_churn_flag"] = (full_probabilities >= 0.5).astype(int)
    full_predictions["prediction_gap"] = full_predictions["predicted_churn_probability"] - frame["renewal_risk_signal"]
    full_predictions = full_predictions.sort_values("predicted_churn_probability", ascending=False)

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": weights,
            "absolute_impact": np.abs(weights),
        }
    ).sort_values("absolute_impact", ascending=False)

    metrics = {
        "accuracy": _accuracy_score(y_test, test_predictions),
        "roc_auc": _roc_auc_score(y_test, test_probabilities),
        "baseline_positive_rate": float(target.mean()),
        "predicted_positive_rate": float(full_predictions["predicted_churn_flag"].mean()),
    }
    return ChurnModelResult(
        predictions=full_predictions,
        feature_importance=feature_importance,
        metrics=metrics,
        weights=weights,
        bias=bias,
        feature_names=feature_names,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
    )


def simulate_churn_scenario(
    database_path: str,
    model_result: ChurnModelResult,
    *,
    resolution_hours_delta: float = 0.0,
    nps_delta: float = 0.0,
    adoption_delta: float = 0.0,
    renewal_window_days_delta: int = 0,
) -> pd.DataFrame:
    scenario_frame = get_churn_model_dataset(database_path).copy()
    scenario_frame["avg_resolution_hours"] = (scenario_frame["avg_resolution_hours"] + resolution_hours_delta).clip(lower=1.0)
    scenario_frame["avg_nps"] = (scenario_frame["avg_nps"] + nps_delta).clip(lower=0.0, upper=10.0)
    scenario_frame["adoption_rate"] = (scenario_frame["adoption_rate"] + adoption_delta).clip(lower=0.05, upper=0.99)
    scenario_frame["renewal_window_days"] = (scenario_frame["renewal_window_days"] + renewal_window_days_delta).clip(lower=15)
    scenario_frame["renewal_risk_signal"] = _compute_renewal_risk_signal(scenario_frame)

    scenario_matrix, _, _, _ = _prepare_churn_model_features(
        scenario_frame,
        feature_names=model_result.feature_names,
        numeric_means=model_result.numeric_means,
        numeric_stds=model_result.numeric_stds,
    )
    scenario_probabilities = _sigmoid(scenario_matrix.to_numpy(dtype=float) @ model_result.weights + model_result.bias)

    scenario_predictions = scenario_frame[["customer_id", "customer_name", "segment", "region", "arr", "renewal_churned"]].copy()
    scenario_predictions["base_churn_probability"] = model_result.predictions.sort_values("customer_id")["predicted_churn_probability"].to_numpy()
    scenario_predictions["scenario_churn_probability"] = scenario_probabilities
    scenario_predictions["probability_change"] = scenario_predictions["scenario_churn_probability"] - scenario_predictions["base_churn_probability"]
    scenario_predictions["scenario_churn_flag"] = (scenario_predictions["scenario_churn_probability"] >= 0.5).astype(int)
    return scenario_predictions.sort_values("probability_change", ascending=False)


def _describe_driver(feature: str, coefficient: float) -> str:
    feature_labels = {
        "renewal_risk_signal": "overall renewal risk signal",
        "renewal_window_days": "renewal timing",
        "avg_resolution_hours": "resolution time",
        "avg_nps": "NPS",
        "adoption_rate": "feature adoption",
        "feature_request_count": "feature-request volume",
        "critical_ticket_count": "critical-ticket volume",
        "ticket_count": "ticket volume",
        "avg_active_users": "active-user depth",
        "avg_actions": "workflow activity",
        "arr": "customer ARR",
        "seats": "seat footprint",
    }

    if feature.startswith("segment_"):
        label = f"segment mix: {feature.split('_', 1)[1]}"
    elif feature.startswith("region_"):
        label = f"regional mix: {feature.split('_', 1)[1]}"
    else:
        label = feature_labels.get(feature, feature.replace("_", " "))

    direction = "raises" if coefficient >= 0 else "reduces"
    return f"{label} {direction} modeled churn pressure"


def _is_controllable_driver(feature: str) -> bool:
    controllable_features = {
        "renewal_risk_signal",
        "renewal_window_days",
        "avg_resolution_hours",
        "avg_nps",
        "adoption_rate",
        "feature_request_count",
        "critical_ticket_count",
        "ticket_count",
        "avg_active_users",
        "avg_actions",
    }
    return feature in controllable_features


def build_scenario_narrative(
    scenario_summary_row: pd.Series | dict[str, Any],
    scenario_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> dict[str, str]:
    summary_row = scenario_summary_row if isinstance(scenario_summary_row, pd.Series) else pd.Series(scenario_summary_row)
    ordered_movers = scenario_predictions.reindex(scenario_predictions["probability_change"].abs().sort_values(ascending=False).index).head(3)
    top_drivers = feature_importance.head(6)

    average_change = float(summary_row["avg_probability_change"])
    risk_direction = "reduced" if average_change < 0 else "increased"
    risk_magnitude = abs(average_change)
    arr_change = float(summary_row["scenario_arr_at_risk"] - summary_row["base_arr_at_risk"])
    account_change = int(summary_row["scenario_high_risk_accounts"] - summary_row["base_high_risk_accounts"])

    summary_text = (
        f"{summary_row['scenario_preset']} {risk_direction} average churn probability by {risk_magnitude:.2%}, "
        f"moving high-risk accounts by {account_change:+d} and ARR at risk by ${arr_change:,.0f}."
    )

    mover_parts: list[str] = []
    for row in ordered_movers.itertuples(index=False):
        direction = "down" if row.probability_change < 0 else "up"
        mover_parts.append(f"{row.customer_name} ({row.segment}) moved {direction} {abs(row.probability_change):.2%}")
    account_text = "Largest account moves: " + "; ".join(mover_parts) + "."

    operational_drivers: list[str] = []
    contextual_drivers: list[str] = []
    for row in top_drivers.itertuples(index=False):
        description = _describe_driver(row.feature, float(row.coefficient))
        if _is_controllable_driver(row.feature) and len(operational_drivers) < 3:
            operational_drivers.append(description)
        elif len(contextual_drivers) < 3:
            contextual_drivers.append(description)

    if not operational_drivers:
        operational_drivers = [_describe_driver(row.feature, float(row.coefficient)) for row in top_drivers.head(3).itertuples(index=False)]

    if not contextual_drivers:
        contextual_drivers = [_describe_driver(row.feature, float(row.coefficient)) for row in top_drivers.tail(min(2, len(top_drivers))).itertuples(index=False)]

    driver_text = "Operational levers: " + "; ".join(operational_drivers) + "."
    context_text = "Contextual factors: " + "; ".join(contextual_drivers) + "."

    return {
        "narrative_summary": summary_text,
        "account_narrative": account_text,
        "driver_narrative": driver_text,
        "context_narrative": context_text,
    }