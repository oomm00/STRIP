"""
STRIP — Analyst Agent

Generates a deterministic structured note from the current market state.
No LLM, no randomness — identical state always produces identical note.
The note is embedded in every TradeObservation before the Trader sees it.
"""

from env.models import TradeObservation


def generate_note(state: TradeObservation) -> str:
    """
    Produce a deterministic analyst note from the current observation.

    The note describes:
    - Price trend direction (upward/downward relative to MA5)
    - Volatility level (high/low based on 3% threshold)
    - Current portfolio exposure
    - Action bias based on conditions

    Returns:
        A plain-English reasoning note (string).
    """
    # Trend direction
    trend = "upward" if state.current_price > state.ma5 else "downward"

    # Volatility classification
    vol_label = "high" if state.avg_volatility > 0.03 else "low"
    vol_pct = state.avg_volatility * 100

    # Portfolio exposure ratio
    if state.portfolio_value > 0:
        exposure = state.holdings * state.current_price / state.portfolio_value
    else:
        exposure = 0.0

    # Determine action bias
    if vol_label == "high":
        bias = "reduce exposure — volatility is elevated"
    elif trend == "upward" and exposure < 0.6:
        bias = "hold or add cautiously — momentum supports continuation"
    elif trend == "downward" and exposure > 0.2:
        bias = "consider reducing — downward pressure is sustained"
    else:
        bias = "hold and monitor — no strong signal"

    return (
        f"Price trend is {trend} over the last 5 steps "
        f"(MA5: {state.ma5:.2f} -> current: {state.current_price:.2f}). "
        f"Volatility is {vol_label} ({vol_pct:.1f}% avg move). "
        f"Current exposure is {exposure:.0%} of portfolio. "
        f"Bias: {bias}."
    )
