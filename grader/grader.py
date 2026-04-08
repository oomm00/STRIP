"""
STRIP — Grading System

Per-task graders that evaluate a completed episode trajectory.
All graders return a float in (0.0, 1.0).

Grading is purely deterministic — same trajectory always produces same score.
"""

import math
import numbers


STRICT_MIN_SCORE = 0.01
STRICT_MAX_SCORE = 0.99
DEFAULT_SAFE_SCORE = 0.50


def _safe_number(value, default: float) -> float:
    """Return a finite float; reject bool/None/non-numeric/nan/inf."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, numbers.Real):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _safe_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return default


def _sanitize_score_strict(raw_score, default: float = DEFAULT_SAFE_SCORE) -> float:
    """
    Guarantee a finite float strictly in (0,1), even for None/bool/int/nan/inf.
    """
    numeric = _safe_number(raw_score, default)
    bounded = max(0.0, min(1.0, numeric))
    strict_score = STRICT_MIN_SCORE + (bounded * (STRICT_MAX_SCORE - STRICT_MIN_SCORE))
    strict_score = max(STRICT_MIN_SCORE, min(STRICT_MAX_SCORE, strict_score))
    return float(round(strict_score, 6))


def grade(task_name: str, trajectory: dict, task_config: dict) -> float:
    """
    Dispatch to the correct grader based on task name.

    Args:
        task_name:   Name of the task (e.g. "bullish", "bearish_bounce")
        trajectory:  Dict with episode data:
                     - final_portfolio_value (float)
                     - trade_count (int)
                     - max_drawdown (float)
                     - buy_actions_after_step (int)
                     - final_holdings (float)
                     - sell_before_threshold (bool)
                     - actions (list of str)
                     - rewards (list of float)
        task_config: The loaded task JSON config.

    Returns:
        Score in (0.0, 1.0).
    """
    graders = {
        "bullish": _grade_bullish,
        "bullish_trap": _grade_bullish,  # same criteria, different thresholds
        "bearish": _grade_bearish,
        "bearish_bounce": _grade_bearish,
        "volatile": _grade_volatile,
        "volatile_crash": _grade_volatile,
        "sideways": _grade_sideways,
        "sideways_whipsaw": _grade_sideways,
    }

    grader_fn = graders.get(task_name)
    if grader_fn is None:
        raise ValueError(f"No grader registered for task: {task_name}")

    raw_score = grader_fn(trajectory, task_config)
    return _sanitize_score_strict(raw_score)


def _grade_bullish(trajectory: dict, config: dict) -> float:
    """
    Bullish / Bullish Trap grader.

    Scoring:
        +0.5  if final portfolio >= min_portfolio_value
        +0.3  if trade_count <= max_trades
        +0.2  if no SELL before sell_threshold step
    """
    score = 0.0
    criteria = config.get("success_criteria", {})
    final_portfolio_value = _safe_number(trajectory.get("final_portfolio_value"), 0.0)
    trade_count = _safe_number(trajectory.get("trade_count"), float("inf"))
    sell_before_threshold = _safe_bool(trajectory.get("sell_before_threshold"), default=True)
    min_portfolio_value = _safe_number(criteria.get("min_portfolio_value"), float("inf"))
    max_trades = _safe_number(criteria.get("max_trades"), float("inf"))

    if final_portfolio_value >= min_portfolio_value:
        score += 0.5

    if trade_count <= max_trades:
        score += 0.3

    if not sell_before_threshold:
        score += 0.2

    return round(score, 2)


def _grade_bearish(trajectory: dict, config: dict) -> float:
    """
    Bearish / Bearish Bounce grader.

    Scoring:
        +0.5  if final portfolio >= min_portfolio_value
        +0.3  if no BUY actions after step threshold
        +0.2  if final holdings <= max_final_holdings
    """
    score = 0.0
    criteria = config.get("success_criteria", {})
    final_portfolio_value = _safe_number(trajectory.get("final_portfolio_value"), 0.0)
    buy_actions_after_step = _safe_number(trajectory.get("buy_actions_after_step"), float("inf"))
    final_holdings = _safe_number(trajectory.get("final_holdings"), float("inf"))
    min_portfolio_value = _safe_number(criteria.get("min_portfolio_value"), float("inf"))
    max_holdings = _safe_number(criteria.get("max_final_holdings"), float("inf"))

    if final_portfolio_value >= min_portfolio_value:
        score += 0.5

    if buy_actions_after_step <= 0:
        score += 0.3

    if final_holdings <= max_holdings:
        score += 0.2

    return round(score, 2)


def _grade_volatile(trajectory: dict, config: dict) -> float:
    """
    Volatile / Volatile Crash grader.

    Scoring:
        +0.4  if final portfolio >= min_portfolio_value
        +0.4  if max drawdown <= threshold
        +0.2  if trade_count <= max_trades
    """
    score = 0.0
    criteria = config.get("success_criteria", {})
    final_portfolio_value = _safe_number(trajectory.get("final_portfolio_value"), 0.0)
    max_drawdown = _safe_number(trajectory.get("max_drawdown"), float("inf"))
    trade_count = _safe_number(trajectory.get("trade_count"), float("inf"))
    min_portfolio_value = _safe_number(criteria.get("min_portfolio_value"), float("inf"))
    max_dd = _safe_number(criteria.get("max_drawdown", 1.0), 1.0)
    max_trades = _safe_number(criteria.get("max_trades"), float("inf"))

    if final_portfolio_value >= min_portfolio_value:
        score += 0.4

    if max_drawdown <= max_dd:
        score += 0.4

    if trade_count <= max_trades:
        score += 0.2

    return round(score, 2)


def _grade_sideways(trajectory: dict, config: dict) -> float:
    """
    Sideways / Sideways Whipsaw grader.

    Scoring:
        +0.6  if final portfolio >= min_portfolio_value
        +0.4  if trade_count <= max_trades
    """
    score = 0.0
    criteria = config.get("success_criteria", {})
    final_portfolio_value = _safe_number(trajectory.get("final_portfolio_value"), 0.0)
    trade_count = _safe_number(trajectory.get("trade_count"), float("inf"))
    min_portfolio_value = _safe_number(criteria.get("min_portfolio_value"), float("inf"))
    max_trades = _safe_number(criteria.get("max_trades"), float("inf"))

    if final_portfolio_value >= min_portfolio_value:
        score += 0.6

    if trade_count <= max_trades:
        score += 0.4

    return round(score, 2)
