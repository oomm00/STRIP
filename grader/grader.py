"""
STRIP — Grading System

Per-task graders that evaluate a completed episode trajectory.
All graders return a float in (0.0, 1.0).

Grading is purely deterministic — same trajectory always produces same score.
"""


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
    
    # Ensure score is strictly between 0 and 1 (not 0.0, not 1.0)
    # Clamp raw_score to [0, 1] first, then map to (0.001, 0.999)
    raw_score = max(0.0, min(1.0, raw_score))
    final_score = 0.001 + (raw_score * 0.998)
    
    # Extra safety: clamp to strict bounds
    final_score = max(0.001, min(0.999, final_score))
    return round(final_score, 4)


def _grade_bullish(trajectory: dict, config: dict) -> float:
    """
    Bullish / Bullish Trap grader.

    Scoring:
        +0.5  if final portfolio >= min_portfolio_value
        +0.3  if trade_count <= max_trades
        +0.2  if no SELL before sell_threshold step
    """
    score = 0.0
    criteria = config["success_criteria"]

    if trajectory["final_portfolio_value"] >= criteria["min_portfolio_value"]:
        score += 0.5

    if trajectory["trade_count"] <= criteria["max_trades"]:
        score += 0.3

    if not trajectory["sell_before_threshold"]:
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
    criteria = config["success_criteria"]

    if trajectory["final_portfolio_value"] >= criteria["min_portfolio_value"]:
        score += 0.5

    if trajectory["buy_actions_after_step"] <= 0:
        score += 0.3

    max_holdings = criteria.get("max_final_holdings", float("inf"))
    if trajectory["final_holdings"] <= max_holdings:
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
    criteria = config["success_criteria"]

    if trajectory["final_portfolio_value"] >= criteria["min_portfolio_value"]:
        score += 0.4

    max_dd = criteria.get("max_drawdown", 1.0)
    if trajectory["max_drawdown"] <= max_dd:
        score += 0.4

    max_trades = criteria.get("max_trades", float("inf"))
    if trajectory["trade_count"] <= max_trades:
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
    criteria = config["success_criteria"]

    if trajectory["final_portfolio_value"] >= criteria["min_portfolio_value"]:
        score += 0.6

    max_trades = criteria.get("max_trades", float("inf"))
    if trajectory["trade_count"] <= max_trades:
        score += 0.4

    return round(score, 2)
