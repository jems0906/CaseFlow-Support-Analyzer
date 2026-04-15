from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


SEED = 42
FEATURE_AREAS = [
    "Reporting",
    "Automation",
    "Knowledge Base",
    "Routing",
    "Integrations",
    "Admin Controls",
]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise", "Strategic"]
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
TICKET_TYPES = ["Incident", "Question", "Feature Request", "Bug"]
SEVERITIES = ["Low", "Medium", "High", "Critical"]
FEEDBACK_THEMES = [
    "Support responsiveness",
    "Missing workflow feature",
    "Reporting depth",
    "Ease of use",
    "Platform stability",
    "Integration coverage",
]
TICKET_TOPIC_LIBRARY = {
    "Reporting": [
        ("dashboard export", "Customer needs more flexible dashboard export and scheduled report delivery."),
        ("report filters", "Teams are struggling to apply saved filters and custom report breakdowns."),
        ("executive visibility", "Leadership wants better roll-up metrics and clearer trend reporting."),
    ],
    "Automation": [
        ("workflow builder", "Agents need more reliable workflow automation and conditional routing logic."),
        ("rule coverage", "Customer is asking for broader automation coverage across escalations and handoffs."),
        ("trigger delays", "Automated workflows are firing late and creating manual follow-up work."),
    ],
    "Knowledge Base": [
        ("search relevance", "Users report weak article search relevance and duplicate content surfacing."),
        ("content governance", "Support leads want stronger article lifecycle controls and approval flows."),
        ("self-service gaps", "Customers are missing guided self-service journeys and article recommendations."),
    ],
    "Routing": [
        ("queue balancing", "Ticket queue balancing is inconsistent across teams and regions."),
        ("assignment rules", "Assignment rules are not capturing product specialists at the right time."),
        ("escalation path", "Escalation path definitions need more flexibility for priority cases."),
    ],
    "Integrations": [
        ("CRM sync", "CRM sync is failing intermittently and leaves support context incomplete."),
        ("API coverage", "Customer is requesting wider API coverage and cleaner webhook events."),
        ("third-party connector", "Connector reliability issues are blocking downstream workflows."),
    ],
    "Admin Controls": [
        ("permission model", "Admins need more granular permission controls for supervisors and vendors."),
        ("audit trail", "Audit trail visibility is not detailed enough for compliance reviews."),
        ("workspace setup", "Workspace setup and configuration changes are too hard to manage safely."),
    ],
}


@dataclass(frozen=True)
class GeneratedData:
    customers: pd.DataFrame
    tickets: pd.DataFrame
    nps_responses: pd.DataFrame
    usage_logs: pd.DataFrame


def _weighted_choice(rng: np.random.Generator, values: list[str], probabilities: list[float], size: int) -> np.ndarray:
    return rng.choice(values, p=probabilities, size=size)


def _build_ticket_text(
    rng: np.random.Generator,
    feature_area: str,
    ticket_type: str,
    severity: str,
) -> tuple[str, str]:
    topic, narrative = TICKET_TOPIC_LIBRARY[feature_area][int(rng.integers(0, len(TICKET_TOPIC_LIBRARY[feature_area])))]
    title = f"{feature_area} {ticket_type.lower()} - {topic}"
    summary = (
        f"{severity} priority {ticket_type.lower()} tied to {feature_area.lower()}: {narrative} "
        f"Support team is tracking customer impact, workaround complexity, and follow-up ownership."
    )
    return title, summary


def generate_mock_data(ticket_count: int = 5_000) -> GeneratedData:
    rng = np.random.default_rng(SEED)
    customer_count = 250
    today = date(2026, 4, 14)
    base_start = today - timedelta(days=540)

    customer_ids = np.arange(1, customer_count + 1)
    segments = _weighted_choice(rng, SEGMENTS, [0.42, 0.28, 0.2, 0.1], customer_count)
    arr_by_segment = {
        "SMB": (12000, 42000),
        "Mid-Market": (45000, 120000),
        "Enterprise": (130000, 350000),
        "Strategic": (360000, 850000),
    }
    seat_by_segment = {
        "SMB": (15, 80),
        "Mid-Market": (81, 300),
        "Enterprise": (301, 1200),
        "Strategic": (1201, 5000),
    }

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_name": [f"Customer {identifier:03d}" for identifier in customer_ids],
            "segment": segments,
            "region": _weighted_choice(rng, REGIONS, [0.4, 0.25, 0.2, 0.15], customer_count),
            "arr": [rng.integers(*arr_by_segment[segment]) for segment in segments],
            "seats": [rng.integers(*seat_by_segment[segment]) for segment in segments],
            "start_date": [base_start + timedelta(days=int(rng.integers(0, 420))) for _ in customer_ids],
        }
    )

    ticket_customer_ids = rng.choice(customer_ids, size=ticket_count, replace=True)
    ticket_segments = customers.set_index("customer_id").loc[ticket_customer_ids, "segment"].to_numpy()
    ticket_feature_area = _weighted_choice(
        rng,
        FEATURE_AREAS,
        [0.18, 0.16, 0.11, 0.19, 0.2, 0.16],
        ticket_count,
    )
    ticket_types = _weighted_choice(rng, TICKET_TYPES, [0.42, 0.22, 0.18, 0.18], ticket_count)
    severities = _weighted_choice(rng, SEVERITIES, [0.34, 0.37, 0.21, 0.08], ticket_count)
    opened_offsets = rng.integers(0, 365, size=ticket_count)
    opened_dates = [today - timedelta(days=int(offset)) for offset in opened_offsets]

    segment_resolution_bias = {
        "SMB": 1.05,
        "Mid-Market": 1.0,
        "Enterprise": 1.18,
        "Strategic": 1.24,
    }
    severity_resolution_bias = {
        "Low": 10,
        "Medium": 22,
        "High": 45,
        "Critical": 78,
    }

    resolution_hours = []
    ticket_titles = []
    ticket_summaries = []
    for segment, severity, ticket_type in zip(ticket_segments, severities, ticket_types, strict=False):
        baseline = severity_resolution_bias[severity] * segment_resolution_bias[segment]
        type_bias = 1.35 if ticket_type == "Bug" else 1.0
        if ticket_type == "Feature Request":
            baseline *= 1.15
        resolution_hours.append(max(2.0, rng.normal(baseline * type_bias, 8.0)))

    for feature_area, ticket_type, severity in zip(ticket_feature_area, ticket_types, severities, strict=False):
        title, summary = _build_ticket_text(rng, feature_area, ticket_type, severity)
        ticket_titles.append(title)
        ticket_summaries.append(summary)

    tickets = pd.DataFrame(
        {
            "ticket_id": np.arange(1, ticket_count + 1),
            "customer_id": ticket_customer_ids,
            "opened_at": opened_dates,
            "resolved_at": [opened + timedelta(hours=float(hours)) for opened, hours in zip(opened_dates, resolution_hours, strict=False)],
            "ticket_type": ticket_types,
            "feature_area": ticket_feature_area,
            "severity": severities,
            "ticket_title": ticket_titles,
            "ticket_summary": ticket_summaries,
            "resolution_hours": np.round(resolution_hours, 2),
        }
    )

    nps_records = []
    for customer in customers.itertuples(index=False):
        months = pd.date_range(today - timedelta(days=330), periods=12, freq="MS")
        segment_bias = {"SMB": 7.6, "Mid-Market": 7.2, "Enterprise": 6.9, "Strategic": 6.8}[customer.segment]
        for month in months:
            usage_factor = rng.normal(0.0, 0.8)
            score = np.clip(rng.normal(segment_bias + usage_factor, 1.6), 0, 10)
            theme = rng.choice(FEEDBACK_THEMES)
            nps_records.append(
                {
                    "response_id": len(nps_records) + 1,
                    "customer_id": customer.customer_id,
                    "response_date": month.date(),
                    "score": round(float(score), 1),
                    "theme": theme,
                    "comment": f"{theme} feedback from {customer.customer_name}. The team noted recurring friction around adoption, support workflow clarity, and product depth.",
                }
            )

    nps_responses = pd.DataFrame(nps_records)

    usage_records = []
    for customer in customers.itertuples(index=False):
        segment_multiplier = {"SMB": 0.92, "Mid-Market": 0.96, "Enterprise": 1.0, "Strategic": 1.03}[customer.segment]
        for month in pd.date_range(today - timedelta(days=330), periods=12, freq="MS"):
            for feature_area in FEATURE_AREAS:
                adoption_base = {
                    "Reporting": 0.78,
                    "Automation": 0.62,
                    "Knowledge Base": 0.58,
                    "Routing": 0.81,
                    "Integrations": 0.49,
                    "Admin Controls": 0.66,
                }[feature_area]
                adoption_score = float(np.clip(rng.normal(adoption_base * segment_multiplier, 0.12), 0.12, 0.98))
                seats_active = max(2, int(customer.seats * adoption_score * rng.uniform(0.35, 0.72)))
                usage_records.append(
                    {
                        "log_id": len(usage_records) + 1,
                        "customer_id": customer.customer_id,
                        "usage_month": month.date(),
                        "feature_area": feature_area,
                        "active_users": seats_active,
                        "sessions": int(seats_active * rng.uniform(3, 15)),
                        "actions": int(seats_active * rng.uniform(20, 120)),
                        "adoption_score": round(adoption_score, 4),
                    }
                )

    usage_logs = pd.DataFrame(usage_records)

    ticket_customer_agg = (
        tickets.groupby("customer_id")
        .agg(
            avg_resolution_hours=("resolution_hours", "mean"),
            ticket_volume=("ticket_id", "count"),
            feature_request_count=("ticket_type", lambda values: int((values == "Feature Request").sum())),
            critical_ticket_count=("severity", lambda values: int((values == "Critical").sum())),
        )
        .reset_index()
    )
    nps_customer_agg = nps_responses.groupby("customer_id").agg(avg_nps=("score", "mean")).reset_index()
    usage_customer_agg = usage_logs.groupby("customer_id").agg(adoption_rate=("adoption_score", "mean")).reset_index()

    churn_frame = (
        customers[["customer_id", "segment"]]
        .merge(ticket_customer_agg, on="customer_id", how="left")
        .merge(nps_customer_agg, on="customer_id", how="left")
        .merge(usage_customer_agg, on="customer_id", how="left")
    )

    segment_risk_bias = {"SMB": 0.05, "Mid-Market": 0.02, "Enterprise": 0.04, "Strategic": 0.06}
    churn_scores = []
    renewal_windows = []
    churn_labels = []
    for row in churn_frame.itertuples(index=False):
        score = (
            0.22 * min(row.avg_resolution_hours / 72.0, 1.5)
            + 0.32 * ((10.0 - row.avg_nps) / 10.0)
            + 0.26 * (1.0 - row.adoption_rate)
            + 0.12 * min(row.feature_request_count / 12.0, 1.2)
            + 0.08 * min(row.critical_ticket_count / 5.0, 1.0)
            + segment_risk_bias[row.segment]
            + float(rng.normal(0.0, 0.04))
        )
        score = float(np.clip(score, 0.02, 0.98))
        renewal_window_days = int(rng.integers(30, 270))
        churned = int(score > 0.54 or (score > 0.47 and renewal_window_days < 120))
        churn_scores.append(round(score, 4))
        renewal_windows.append(renewal_window_days)
        churn_labels.append(churned)

    customers = customers.merge(
        pd.DataFrame(
            {
                "customer_id": churn_frame["customer_id"],
                "renewal_risk_signal": churn_scores,
                "renewal_window_days": renewal_windows,
                "renewal_churned": churn_labels,
            }
        ),
        on="customer_id",
        how="left",
    )

    return GeneratedData(
        customers=customers,
        tickets=tickets,
        nps_responses=nps_responses,
        usage_logs=usage_logs,
    )
