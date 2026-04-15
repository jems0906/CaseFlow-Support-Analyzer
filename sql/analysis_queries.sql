-- name: kpis
WITH ticket_summary AS (
    SELECT COUNT(*) AS ticket_count, AVG(resolution_hours) AS avg_resolution_hours
    FROM tickets
),
nps_summary AS (
    SELECT AVG(score) AS avg_nps
    FROM nps_responses
),
customer_summary AS (
    SELECT COUNT(*) AS customer_count
    FROM customers
),
churn_summary AS (
    SELECT SUM(CASE WHEN churn_risk_score >= 0.40 THEN 1 ELSE 0 END) AS high_risk_customers
    FROM customer_churn_features
)
SELECT
    customer_summary.customer_count,
    ticket_summary.ticket_count,
    ROUND(ticket_summary.avg_resolution_hours, 2) AS avg_resolution_hours,
    ROUND(nps_summary.avg_nps, 2) AS avg_nps,
    churn_summary.high_risk_customers
FROM customer_summary
CROSS JOIN ticket_summary
CROSS JOIN nps_summary
CROSS JOIN churn_summary;

-- name: feature_gap
WITH ticket_agg AS (
    SELECT
        feature_area,
        COUNT(*) AS ticket_count,
        AVG(resolution_hours) AS avg_resolution_hours
    FROM tickets
    GROUP BY feature_area
),
usage_agg AS (
    SELECT
        feature_area,
        AVG(adoption_score) AS adoption_rate
    FROM usage_logs
    GROUP BY feature_area
),
ticket_total AS (
    SELECT COUNT(*) AS total_tickets
    FROM tickets
)
SELECT
    ticket_agg.feature_area,
    ROUND(ticket_agg.ticket_count * 1.0 / ticket_total.total_tickets, 4) AS support_ticket_share,
    ROUND(ticket_agg.avg_resolution_hours, 2) AS avg_resolution_hours,
    ROUND(usage_agg.adoption_rate, 4) AS adoption_rate,
    ROUND((ticket_agg.ticket_count * 1.0 / ticket_total.total_tickets) * (1.0 - usage_agg.adoption_rate) * 100, 2) AS gap_score
FROM ticket_agg
JOIN usage_agg ON usage_agg.feature_area = ticket_agg.feature_area
CROSS JOIN ticket_total
ORDER BY gap_score DESC;

-- name: nps_trends
SELECT
    c.segment,
    substr(n.response_date, 1, 7) AS response_month,
    ROUND(AVG(n.score), 2) AS avg_nps,
    COUNT(*) AS responses
FROM nps_responses n
JOIN customers c ON c.customer_id = n.customer_id
GROUP BY c.segment, substr(n.response_date, 1, 7)
ORDER BY response_month, c.segment;

-- name: churn_prediction
SELECT
    c.customer_id,
    c.customer_name,
    c.segment,
    c.arr,
    churn.avg_resolution_hours,
    churn.avg_nps,
    churn.adoption_rate,
    churn.churn_risk_score
FROM customers c
JOIN customer_churn_features churn ON churn.customer_id = c.customer_id
ORDER BY churn.churn_risk_score DESC;

-- name: feature_requests
WITH request_base AS (
    SELECT
        t.feature_area,
        t.ticket_id,
        t.customer_id,
        c.arr
    FROM tickets t
    JOIN customers c ON c.customer_id = t.customer_id
    WHERE t.ticket_type = 'Feature Request'
),
nps_agg AS (
    SELECT
        customer_id,
        AVG(10.0 - score) AS dissatisfaction_signal
    FROM nps_responses
    GROUP BY customer_id
),
request_customer_agg AS (
    SELECT
        feature_area,
        customer_id,
        COUNT(ticket_id) AS request_count,
        MAX(arr) AS customer_arr
    FROM request_base
    GROUP BY feature_area, customer_id
)
SELECT
    request_customer_agg.feature_area,
    SUM(request_customer_agg.request_count) AS frequency,
    COUNT(request_customer_agg.customer_id) AS impacted_customers,
    ROUND(SUM(request_customer_agg.customer_arr), 2) AS impacted_arr,
    ROUND(AVG(nps_agg.dissatisfaction_signal), 2) AS dissatisfaction_signal,
    ROUND(
        SUM(request_customer_agg.request_count) * 0.45
        + COUNT(request_customer_agg.customer_id) * 0.25
        + SUM(request_customer_agg.customer_arr) / 10000.0 * 0.20
        + AVG(nps_agg.dissatisfaction_signal) * 0.10,
        2
    ) AS priority_score
FROM request_customer_agg
JOIN nps_agg ON nps_agg.customer_id = request_customer_agg.customer_id
GROUP BY request_customer_agg.feature_area
ORDER BY priority_score DESC;
