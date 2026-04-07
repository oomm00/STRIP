"""
STRIP — Reward Engine

Implements the full reward computation:
  R_t = delta_V_t - 1.0 * C_t - 0.5 * risk_t - 2.0 * E_t

All constants are fixed across tasks:
  alpha = 1.0  (transaction cost weight)
  beta  = 0.5  (position risk weight)
  gamma = 2.0  (early exit penalty weight)
"""

from env.models import TradeAction, TradeReward


# --- Fixed reward constants ---
ALPHA = 1.0   # transaction cost weight
BETA = 0.5    # position risk weight
GAMMA = 2.0   # early exit penalty weight


def normalize_reward(raw_reward: float, initial_capital: float) -> float:
    """
    Normalise a raw reward into the [0.0, 1.0] range.

    Uses a per-step scale (5% of initial capital) to ensure step-level
    reward variations are meaningful, not crushed near 0.5.

    Clips to [-scale, +scale] then linearly maps:
        -scale → 0.0
        0      → 0.5
        +scale → 1.0
    """
    scale = initial_capital * 0.05  # per-step scale = 5% of capital
    clipped = max(-scale, min(scale, raw_reward))
    return (clipped + scale) / (2 * scale)


def compute_reward(
    prev_cash: float,
    prev_holdings: float,
    prev_price: float,
    curr_cash: float,
    curr_holdings: float,
    curr_price: float,
    trade_value: float,
    action: TradeAction,
    step: int,
    task_config: dict,
) -> TradeReward:
    """
    Compute the full decomposed reward for a single step.

    Args:
        prev_cash:     cash before this step
        prev_holdings: holdings before this step
        prev_price:    price before this step
        curr_cash:     cash after this step
        curr_holdings: holdings after this step
        curr_price:    price after this step (new price revealed)
        trade_value:   absolute value of the trade executed this step
        action:        the action taken
        step:          current step index (0-based)
        task_config:   loaded task JSON with sell_threshold, initial_capital, etc.

    Returns:
        TradeReward with all components filled.
    """
    # Portfolio values
    v_prev = prev_cash + prev_holdings * prev_price
    v_curr = curr_cash + curr_holdings * curr_price
    delta_v = v_curr - v_prev

    # Transaction cost: |trade_value| * 0.1%
    c_t = abs(trade_value) * task_config.get("transaction_cost_rate", 0.001)

    # Position risk: holdings * |price_change| / prev_price
    if prev_price > 0:
        sigma_t = abs(curr_price - prev_price) / prev_price
    else:
        sigma_t = 0.0
    risk_t = curr_holdings * sigma_t

    # Early exit penalty
    sell_threshold = task_config.get("sell_threshold", 0)
    e_t = 1.0 if (action == TradeAction.SELL and step < sell_threshold) else 0.0

    # Terminal bonus (applied externally at episode end — set to 0 during step)
    terminal_bonus = 0.0

    # Raw reward
    raw = delta_v - ALPHA * c_t - BETA * risk_t - GAMMA * e_t

    # Normalised reward
    initial_capital = task_config["initial_capital"]
    normalized = normalize_reward(raw, initial_capital)

    return TradeReward(
        raw_reward=round(raw, 6),
        normalized_reward=round(normalized, 6),
        delta_v=round(delta_v, 6),
        transaction_cost=round(c_t, 6),
        position_risk=round(risk_t, 6),
        early_exit_penalty=round(e_t, 6),
        terminal_bonus=terminal_bonus,
    )


def apply_terminal_bonus(
    reward: TradeReward,
    success: bool,
    task_config: dict,
) -> TradeReward:
    """
    Apply terminal bonus (+10 success / -10 failure) to the final step reward.
    Returns a new TradeReward with updated values.
    """
    bonus = 10.0 if success else -10.0
    raw = reward.raw_reward + bonus
    normalized = normalize_reward(raw, task_config["initial_capital"])

    return TradeReward(
        raw_reward=round(raw, 6),
        normalized_reward=round(normalized, 6),
        delta_v=reward.delta_v,
        transaction_cost=reward.transaction_cost,
        position_risk=reward.position_risk,
        early_exit_penalty=reward.early_exit_penalty,
        terminal_bonus=bonus,
    )


def success_criteria_met(
    final_portfolio_value: float,
    trade_count: int,
    max_drawdown: float,
    buy_actions_after_step: int,
    final_holdings: float,
    sell_before_threshold: bool,
    task_config: dict,
) -> bool:
    """
    Check if the episode meets the task-specific success criteria.

    This is a simplified binary check — detailed scoring is in grader.py.
    """
    criteria = task_config["success_criteria"]

    # Portfolio value check
    min_value = criteria.get("min_portfolio_value", 0)
    if final_portfolio_value < min_value:
        return False

    # Trade count check
    max_trades = criteria.get("max_trades")
    if max_trades is not None and trade_count > max_trades:
        return False

    # Max drawdown check
    max_dd = criteria.get("max_drawdown")
    if max_dd is not None and max_drawdown > max_dd:
        return False

    # No sell before step check
    if criteria.get("no_sell_before_step") and sell_before_threshold:
        return False

    # Buy restrictions
    buy_restriction = criteria.get("max_buys_after_step")
    if buy_restriction is not None:
        if buy_actions_after_step > buy_restriction.get("max_buys", 0):
            return False

    # Final holdings check
    max_holdings = criteria.get("max_final_holdings")
    if max_holdings is not None and final_holdings > max_holdings:
        return False

    return True
