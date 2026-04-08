"""
STRIP — Trader Agent (Rule-Based Baseline)

A simple rule-based offline baseline that reads the analyst_note
to choose an action. In inference.py, this logic is replaced by
an LLM call via the OpenAI client.
"""

from env.models import TradeObservation, TradeAction

# Simple action memory for stability (prevents rapid flipping)
_last_actions: list = []


def _get_price_trend(obs: TradeObservation) -> str:
    """Analyze recent price movement."""
    if len(obs.price_history) < 3:
        return "neutral"
    recent = obs.price_history[-3:]
    if recent[-1] > recent[0] * 1.02:
        return "up"
    elif recent[-1] < recent[0] * 0.98:
        return "down"
    return "neutral"


def _portfolio_ratio(obs: TradeObservation) -> float:
    """Return ratio of holdings value to total portfolio (0.0 to 1.0)."""
    if obs.portfolio_value <= 0:
        return 0.0
    holdings_value = obs.holdings * obs.current_price
    return holdings_value / obs.portfolio_value


def _is_price_stable(obs: TradeObservation) -> bool:
    """Check if price movement is minimal (sideways behavior)."""
    if len(obs.price_history) < 3:
        return True
    recent = obs.price_history[-3:]
    change = abs(recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0
    return change < 0.015  # Less than 1.5% change = stable


def choose_action(obs: TradeObservation) -> TradeAction:
    global _last_actions
    import random

    note = obs.analyst_note.lower()
    scenario = obs.scenario_label.lower()
    trend = _get_price_trend(obs)
    ratio = _portfolio_ratio(obs)
    stable = _is_price_stable(obs)
    last_action = _last_actions[-1] if _last_actions else None
    prev_action = _last_actions[-2] if len(_last_actions) >= 2 else None

    strong_sell = ("reduc" in note or "risk" in note) and obs.holdings > 0
    strong_buy = ("momentum" in note or "opportun" in note) and obs.holdings == 0 and obs.cash > 0

    # Time-based activity: more active early/mid, conservative late
    progress = obs.step / obs.max_steps if obs.max_steps > 0 else 0
    early = progress < 0.3
    mid = 0.3 <= progress < 0.7
    late = progress >= 0.7

    action = TradeAction.HOLD

    if "sideways" in scenario:
        # Sideways: mostly HOLD, occasional probe (~8%)
        if stable:
            if random.random() < 0.08 and mid:
                if obs.cash > 0 and ratio < 0.5:
                    action = TradeAction.BUY
                elif obs.holdings > 0 and ratio > 0.5:
                    action = TradeAction.SELL
                else:
                    action = TradeAction.HOLD
            else:
                action = TradeAction.HOLD
        elif obs.holdings > 0 and trend == "down" and ratio > 0.8:
            action = TradeAction.SELL
        else:
            action = TradeAction.HOLD

    elif strong_sell:
        action = TradeAction.SELL
    elif strong_buy:
        action = TradeAction.BUY if (trend != "down" or early) else TradeAction.HOLD

    elif "bullish" in scenario:
        # Bullish: BUY early, occasional add mid, conservative late
        if obs.holdings == 0 and obs.cash > 0 and trend != "down":
            if early:
                action = TradeAction.BUY
            elif mid and random.random() < 0.12:
                action = TradeAction.BUY
            else:
                action = TradeAction.HOLD
        elif late and obs.holdings > 0 and trend == "down" and random.random() < 0.15:
            action = TradeAction.SELL
        else:
            action = TradeAction.HOLD

    elif "bearish" in scenario:
        # Bearish: SELL early, maybe delayed sell mid
        if obs.holdings > 0:
            if early:
                action = TradeAction.SELL
            elif mid and trend == "down" and random.random() < 0.10:
                action = TradeAction.SELL
            else:
                action = TradeAction.HOLD
        else:
            action = TradeAction.HOLD

    elif "volatile" in scenario:
        # Volatile: cautious with occasional activity
        if ratio > 0.7 and trend == "down":
            action = TradeAction.SELL
        elif ratio < 0.3 and trend == "up" and obs.cash > 0:
            action = TradeAction.BUY
        elif mid and random.random() < 0.06:
            if obs.cash > 0 and ratio < 0.5:
                action = TradeAction.BUY
            elif obs.holdings > 0 and ratio > 0.5:
                action = TradeAction.SELL
            else:
                action = TradeAction.HOLD
        else:
            action = TradeAction.HOLD

    # Prevent rapid flipping (keep stability)
    if last_action == TradeAction.BUY and action == TradeAction.SELL:
        if not (strong_sell and trend == "down"):
            action = TradeAction.HOLD
    if last_action == TradeAction.SELL and action == TradeAction.BUY:
        if not (strong_buy and trend == "up"):
            action = TradeAction.HOLD
    if prev_action == TradeAction.BUY and last_action == TradeAction.SELL and action == TradeAction.BUY:
        action = TradeAction.HOLD
    if prev_action == TradeAction.SELL and last_action == TradeAction.BUY and action == TradeAction.SELL:
        action = TradeAction.HOLD

    _last_actions.append(action)
    if len(_last_actions) > 2:
        _last_actions.pop(0)
    return action


def reset_action_memory():
    """Reset action memory between tasks."""
    global _last_actions
    _last_actions = []
