from __future__ import annotations

import html

import pandas as pd


def build_executive_summary(
    kpis: pd.Series,
    feature_gap: pd.DataFrame,
    churn: pd.DataFrame,
    feature_requests: pd.DataFrame,
 ) -> list[str]:
    top_gap = feature_gap.iloc[0]
    top_churn = churn.sort_values("churn_risk_score", ascending=False).iloc[0]
    top_request = feature_requests.iloc[0]

    return [
        f"Support operations handled {int(kpis['ticket_count']):,} tickets with an average resolution time of {kpis['avg_resolution_hours']:.1f} hours.",
        f"{top_gap['feature_area']} shows the largest feature gap with {top_gap['support_ticket_share']:.1%} of support volume and only {top_gap['adoption_rate']:.0%} adoption.",
        f"{int(kpis['high_risk_customers'])} customers are in the high-risk churn cohort; {top_churn['customer_name']} currently leads at a risk score of {top_churn['churn_risk_score']:.2f}.",
        f"Top roadmap candidate: {top_request['feature_area']} with priority score {top_request['priority_score']:.2f} and ${top_request['impacted_arr']:,.0f} ARR exposed.",
    ]


def build_executive_report(
    kpis: pd.Series,
    summary: list[str],
    feature_gap: pd.DataFrame,
    churn: pd.DataFrame,
    feature_requests: pd.DataFrame,
    selected_segments: list[str],
    scenario_summary: pd.DataFrame | None = None,
    scenario_history: pd.DataFrame | None = None,
) -> str:
    top_gap = feature_gap.iloc[0]
    top_request = feature_requests.iloc[0]
    at_risk = churn.sort_values("churn_risk_score", ascending=False).head(5)
    segment_scope = ", ".join(selected_segments) if selected_segments else "All segments"

    lines = [
        "# CaseFlow Support Analyzer Executive Report",
        "",
        f"Segment scope: {segment_scope}",
        "",
        "## KPI snapshot",
        f"- Customers analyzed: {int(kpis['customer_count']):,}",
        f"- Tickets analyzed: {int(kpis['ticket_count']):,}",
        f"- Average resolution time: {kpis['avg_resolution_hours']:.1f} hours",
        f"- Average NPS: {kpis['avg_nps']:.1f}",
        f"- High churn risk customers: {int(kpis['high_risk_customers']):,}",
        "",
        "## Executive summary",
    ]
    lines.extend([f"- {item}" for item in summary])
    lines.extend(
        [
            "",
            "## Key findings",
            f"- Largest product gap: {top_gap['feature_area']} with gap score {top_gap['gap_score']:.2f}",
            f"- Highest roadmap priority: {top_request['feature_area']} with score {top_request['priority_score']:.2f}",
            "",
            "## Top churn-risk accounts",
        ]
    )

    for row in at_risk.itertuples(index=False):
        lines.append(
            f"- {row.customer_name} ({row.segment}) | risk {row.churn_risk_score:.2f} | resolution {row.avg_resolution_hours:.1f} hrs | NPS {row.avg_nps:.1f} | adoption {row.adoption_rate:.0%}"
        )

    if scenario_summary is not None and not scenario_summary.empty:
        current_scenario = scenario_summary.iloc[0]
        lines.extend(
            [
                "",
                "## Current Scenario Commentary",
                f"- Playbook: {current_scenario['scenario_preset']}",
                f"- Scope: {current_scenario['segment_scope']}",
                f"- {current_scenario['narrative_summary']}",
                f"- {current_scenario['account_narrative']}",
                f"- {current_scenario['driver_narrative']}",
                f"- {current_scenario['context_narrative']}",
                f"- Inputs: resolution {current_scenario['resolution_hours_delta']:+.0f} hrs, NPS {current_scenario['nps_delta']:+.1f}, adoption {current_scenario['adoption_delta']:+.2f}, renewal window {current_scenario['renewal_window_days_delta']:+.0f} days",
            ]
        )

    if scenario_history is not None and not scenario_history.empty:
        lines.extend(["", "## Saved Scenario Snapshots"])
        for row in scenario_history.sort_values("scenario_id").itertuples(index=False):
            lines.extend(
                [
                    f"### Scenario {row.scenario_id}: {row.scenario_preset}",
                    f"- Scope: {row.segment_scope}",
                    f"- {row.narrative_summary}",
                    f"- {row.account_narrative}",
                    f"- {row.driver_narrative}",
                    f"- {row.context_narrative}",
                ]
            )

    return "\n".join(lines)


def build_board_report_html(
    kpis: pd.Series,
    summary: list[str],
    feature_gap: pd.DataFrame,
    churn: pd.DataFrame,
    feature_requests: pd.DataFrame,
    selected_segments: list[str],
    scenario_summary: pd.DataFrame | None = None,
    scenario_history: pd.DataFrame | None = None,
) -> str:
    segment_scope = ", ".join(selected_segments) if selected_segments else "All segments"
    top_gap = feature_gap.iloc[0]
    top_request = feature_requests.iloc[0]
    top_accounts = churn.sort_values("churn_risk_score", ascending=False).head(5)

    summary_items = "".join(f"<li>{html.escape(item)}</li>" for item in summary)
    top_account_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row.customer_name)}</td>"
        f"<td>{html.escape(row.segment)}</td>"
        f"<td>{row.churn_risk_score:.2f}</td>"
        f"<td>{row.avg_resolution_hours:.1f} hrs</td>"
        f"<td>{row.avg_nps:.1f}</td>"
        f"<td>{row.adoption_rate:.0%}</td>"
        "</tr>"
        for row in top_accounts.itertuples(index=False)
    )

    scenario_section = ""
    if scenario_summary is not None and not scenario_summary.empty:
        current = scenario_summary.iloc[0]
        scenario_section = f"""
        <section class=\"panel\">
          <h2>Current Scenario</h2>
          <div class=\"scenario-meta\">Playbook: {html.escape(str(current['scenario_preset']))} | Scope: {html.escape(str(current['segment_scope']))}</div>
          <ul class=\"scenario-list\">
            <li>{html.escape(str(current['narrative_summary']))}</li>
            <li>{html.escape(str(current['account_narrative']))}</li>
            <li>{html.escape(str(current['driver_narrative']))}</li>
            <li>{html.escape(str(current['context_narrative']))}</li>
          </ul>
          <div class=\"scenario-inputs\">Inputs: resolution {current['resolution_hours_delta']:+.0f} hrs, NPS {current['nps_delta']:+.1f}, adoption {current['adoption_delta']:+.2f}, renewal window {current['renewal_window_days_delta']:+.0f} days</div>
        </section>
        """

    saved_scenarios = ""
    if scenario_history is not None and not scenario_history.empty:
        cards: list[str] = []
        for row in scenario_history.sort_values("scenario_id").itertuples(index=False):
            cards.append(
                f"""
                <article class=\"snapshot-card\">
                  <h3>Scenario {row.scenario_id}: {html.escape(str(row.scenario_preset))}</h3>
                  <p class=\"snapshot-scope\">{html.escape(str(row.segment_scope))}</p>
                  <p>{html.escape(str(row.narrative_summary))}</p>
                  <p>{html.escape(str(row.account_narrative))}</p>
                  <p>{html.escape(str(row.driver_narrative))}</p>
                  <p>{html.escape(str(row.context_narrative))}</p>
                </article>
                """
            )
        saved_scenarios = f"""
        <section class=\"panel\">
          <h2>Saved Scenario Snapshots</h2>
          <div class=\"snapshot-grid\">{''.join(cards)}</div>
        </section>
        """

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>CaseFlow Support Analyzer Board Report</title>
  <style>
    :root {{
      --ink: #183153;
      --muted: #58708a;
      --accent: #0b6e6e;
      --accent-soft: #dff2ef;
      --panel: #ffffff;
      --canvas: #f4f7f8;
      --border: #d4dce3;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", Calibri, sans-serif; color: var(--ink); background: var(--canvas); }}
    .page {{ max-width: 1120px; margin: 0 auto; padding: 32px; }}
    .hero {{ background: linear-gradient(135deg, #113b59 0%, #0b6e6e 100%); color: white; padding: 28px 30px; border-radius: 20px; margin-bottom: 22px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 32px; }}
    .hero p {{ margin: 0; font-size: 16px; opacity: 0.92; }}
    .scope {{ margin-top: 12px; font-size: 14px; opacity: 0.9; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 22px; }}
    .kpi-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 16px; box-shadow: 0 10px 28px rgba(17, 59, 89, 0.07); }}
    .kpi-card .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 8px; }}
    .kpi-card .value {{ font-size: 26px; font-weight: 700; }}
    .layout {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 20px; margin-bottom: 18px; box-shadow: 0 10px 28px rgba(17, 59, 89, 0.07); }}
    .panel h2 {{ margin: 0 0 14px; font-size: 20px; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .highlight {{ background: var(--accent-soft); border-radius: 14px; padding: 14px 16px; margin-bottom: 14px; }}
    .highlight strong {{ display: block; margin-bottom: 4px; }}
    .scenario-meta, .scenario-inputs, .snapshot-scope {{ color: var(--muted); font-size: 13px; margin-bottom: 10px; }}
    .scenario-list {{ padding-left: 18px; }}
    .snapshot-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
    .snapshot-card {{ border: 1px solid var(--border); border-radius: 14px; padding: 14px; background: #fbfcfc; }}
    .snapshot-card h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .snapshot-card p {{ margin: 0 0 8px; font-size: 14px; }}
    @media print {{
      body {{ background: white; }}
      .page {{ max-width: none; padding: 12mm; }}
      .panel, .kpi-card, .hero {{ box-shadow: none; }}
    }}
    @media (max-width: 900px) {{
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .layout {{ grid-template-columns: 1fr; }}
      .snapshot-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class=\"page\">
    <section class=\"hero\">
      <h1>CaseFlow Support Analyzer</h1>
      <p>Board-ready support, product, and churn synthesis for executive review.</p>
      <div class=\"scope\">Segment scope: {html.escape(segment_scope)}</div>
    </section>

    <section class=\"kpi-grid\">
      <div class=\"kpi-card\"><div class=\"label\">Customers</div><div class=\"value\">{int(kpis['customer_count']):,}</div></div>
      <div class=\"kpi-card\"><div class=\"label\">Tickets</div><div class=\"value\">{int(kpis['ticket_count']):,}</div></div>
      <div class=\"kpi-card\"><div class=\"label\">Avg Resolution</div><div class=\"value\">{kpis['avg_resolution_hours']:.1f} hrs</div></div>
      <div class=\"kpi-card\"><div class=\"label\">Average NPS</div><div class=\"value\">{kpis['avg_nps']:.1f}</div></div>
      <div class=\"kpi-card\"><div class=\"label\">High Churn Risk</div><div class=\"value\">{int(kpis['high_risk_customers']):,}</div></div>
    </section>

    <section class=\"layout\">
      <div>
        <section class=\"panel\">
          <h2>Executive Summary</h2>
          <ul>{summary_items}</ul>
        </section>
        <section class=\"panel\">
          <h2>Priority Signals</h2>
          <div class=\"highlight\"><strong>Largest Feature Gap</strong>{html.escape(str(top_gap['feature_area']))} with gap score {top_gap['gap_score']:.2f} and adoption at {top_gap['adoption_rate']:.0%}.</div>
          <div class=\"highlight\"><strong>Top Roadmap Priority</strong>{html.escape(str(top_request['feature_area']))} with priority score {top_request['priority_score']:.2f} and ${top_request['impacted_arr']:,.0f} ARR exposed.</div>
        </section>
        {scenario_section}
        {saved_scenarios}
      </div>
      <div>
        <section class=\"panel\">
          <h2>Top Churn-Risk Accounts</h2>
          <table>
            <thead>
              <tr><th>Customer</th><th>Segment</th><th>Risk</th><th>Resolution</th><th>NPS</th><th>Adoption</th></tr>
            </thead>
            <tbody>{top_account_rows}</tbody>
          </table>
        </section>
      </div>
    </section>
  </div>
</body>
</html>
"""