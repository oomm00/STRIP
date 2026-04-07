"""
STRIP — Trader Agent (Rule-Based Baseline)

A simple rule-based offline baseline that reads the analyst_note
to choose an action. In inference.py, this logic is replaced by
an LLM call via the OpenAI client.
"""

from env.models import TradeObservation, TradeAction


def choose_action(obs: TradeObservation) -> TradeAction:
    """
    Choose a trading action based on the analyst note and portfolio state.

    Decision logic:
    1. "reduce" in note → SELL (high volatility or downtrend with exposure)
    2. "momentum" in note AND no holdings AND cash > 0 → BUY
    3. "momentum" in note AND has holdings → HOLD (already positioned)
    4. Default → HOLD

    Args:
        obs: Current TradeObservation with analyst_note populated.

    Returns:
        TradeAction (BUY, SELL, or HOLD).
    """
    note = obs.analyst_note.lower()

    # Priority 1: reduce exposure signals
    if "reduc" in note:
        if obs.holdings > 0:
            return TradeAction.SELL
        return TradeAction.HOLD

    # Priority 2: momentum signals — buy if not yet positioned
    if "momentum" in note:
        if obs.holdings == 0 and obs.cash > 0:
            return TradeAction.BUY
        return TradeAction.HOLD

    # Default: hold
    return TradeAction.HOLD
