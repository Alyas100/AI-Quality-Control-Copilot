"""
Financial Impact Calculator
================================================================================
Translates predicted FFA degradation into estimated revenue lost per ton.
Business value layer for hackathon judges — ties ML metrics to bottom-line profitability.
"""

FFA_THRESHOLD = 2.5        # % FFA before penalties kick in
PENALTY_PER_01_PCT = 15.0  # $ per 0.1% FFA above threshold
CPO_BASE_PRICE = 800.0     # $/ton
DEFAULT_BATCH_SIZE = 20    # tons

GRADE_MULTIPLIERS = {
    "premium":  {"label": "Premium Grade",  "multiplier": 1.08, "max_ffa": 2.0, "color": "green"},
    "standard": {"label": "Standard Grade", "multiplier": 1.00, "max_ffa": 2.5, "color": "slate"},
    "discount": {"label": "Discount Grade", "multiplier": 0.92, "max_ffa": float("inf"), "color": "red"},
}


def classify_grade(ffa_pct: float) -> dict:
    for grade_key, info in GRADE_MULTIPLIERS.items():
        if ffa_pct <= info["max_ffa"]:
            return {"key": grade_key, **info}
    return {"key": "discount", **GRADE_MULTIPLIERS["discount"]}


def calculate_batch_impact(
    predicted_ffa: float,
    batch_size: float = DEFAULT_BATCH_SIZE,
    cpo_price: float = CPO_BASE_PRICE,
) -> dict:
    ffa_excess = max(0.0, predicted_ffa - FFA_THRESHOLD)
    penalty_per_ton = (ffa_excess / 0.1) * PENALTY_PER_01_PCT
    base_revenue = batch_size * cpo_price
    penalty_total = penalty_per_ton * batch_size
    net_revenue = base_revenue - penalty_total
    revenue_loss_pct = (penalty_total / base_revenue * 100) if base_revenue else 0.0
    grade = classify_grade(predicted_ffa)

    if ffa_excess == 0:
        color = "green"
    elif ffa_excess < 1.0:
        color = "amber"
    else:
        color = "red"

    return {
        "ffa_excess": round(ffa_excess, 2),
        "penalty_per_ton": round(penalty_per_ton, 2),
        "base_revenue": round(base_revenue, 2),
        "penalty_total": round(penalty_total, 2),
        "net_revenue": round(net_revenue, 2),
        "revenue_loss_pct": round(revenue_loss_pct, 2),
        "grade": grade,
        "risk_impact_color": color,
        "batch_size": batch_size,
        "cpo_price": cpo_price,
    }


def calculate_annual_estimate(
    batch_impact: dict,
    daily_batches: int = 3,
    working_days: int = 350,
) -> dict:
    annual_penalty = batch_impact["penalty_total"] * daily_batches * working_days
    annual_revenue = batch_impact["base_revenue"] * daily_batches * working_days
    return {
        "annual_penalty": round(annual_penalty, 2),
        "annual_revenue": round(annual_revenue, 2),
        "annual_loss_pct": round((annual_penalty / annual_revenue * 100) if annual_revenue else 0, 2),
        "daily_batches": daily_batches,
        "working_days": working_days,
    }
