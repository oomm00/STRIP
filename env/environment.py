"""
STRIP — Environment Core

OpenEnv-compliant environment implementing:
- reset(task)  → load task config, initialise state, return first observation
- step(action) → advance market, update portfolio, compute reward, check done
- state()      → return current state snapshot
- close()      → clean up resources
"""

import json
import os
from typing import Tuple

from env.models import TradeAction, TradeObservation, TradeReward
from env.reward import compute_reward, apply_terminal_bonus, success_criteria_met
from agents.analyst import generate_note


# Directory containing task JSON files
TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")


def _load_task_config(task_name: str) -> dict:
    """Load task JSON from the tasks/ directory."""
    path = os.path.join(TASKS_DIR, f"{task_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task config not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def check_done(
    cash: float,
    holdings: float,
    current_price: float,
    step: int,
    config: dict,
) -> bool:
    """
    Check the three termination conditions:
    1. Max steps reached
    2. Capital collapse: portfolio below 80% of initial
    3. No active position: cash = 0 AND holdings = 0
    """
    # 1. Max steps
    if step >= config["max_steps"]:
        return True

    # 2. Capital collapse
    portfolio_value = cash + holdings * current_price
    if portfolio_value <= 0.8 * config["initial_capital"]:
        return True

    # 3. No active position
    if cash == 0 and holdings == 0:
        return True

    return False


class STRIPEnv:
    """
    STRIP (Systematic Trading Reasoning Intelligence Platform) environment.

    OpenEnv-compliant interface with reset(), step(), state(), close().
    """

    def __init__(self):
        self.config = None
        self.price_series = []
        self.current_step = 0
        self.cash = 0.0
        self.holdings = 0.0
        self.current_price = 0.0
        self.scenario_label = ""
        self.done = False
        self.last_action = TradeAction.HOLD
        self.last_trade_value = 0.0

        # Tracking for grading
        self.trade_count = 0
        self.max_portfolio_value = 0.0
        self.max_drawdown = 0.0
        self.buy_actions_after_step = 0
        self.sell_before_threshold = False
        self.rewards_history = []
        self.actions_history = []

    def reset(self, task: str = "bullish") -> TradeObservation:
        """
        Load task config, initialise state, return the first observation.

        Args:
            task: Task name (e.g. "bullish", "bearish_bounce", etc.)

        Returns:
            Initial TradeObservation with analyst note populated.
        """
        self.config = _load_task_config(task)
        self.price_series = self.config["price_series"]
        self.current_step = 0
        self.cash = self.config["initial_cash"]
        self.holdings = self.config["initial_holdings"]
        self.current_price = self.price_series[0]
        self.scenario_label = self.config["scenario_label"]
        self.done = False
        self.last_action = TradeAction.HOLD
        self.last_trade_value = 0.0

        # Reset tracking
        self.trade_count = 0
        portfolio_value = self.cash + self.holdings * self.current_price
        self.max_portfolio_value = portfolio_value
        self.max_drawdown = 0.0
        self.buy_actions_after_step = 0
        self.sell_before_threshold = False
        self.rewards_history = []
        self.actions_history = []

        return self._build_observation()

    def step(self, action: TradeAction) -> Tuple[TradeObservation, float, bool]:
        """
        Execute one environment step.

        1. Save previous state
        2. Execute the trade action
        3. Advance to next price
        4. Compute reward
        5. Check termination
        6. Build observation with analyst note

        Args:
            action: TradeAction (BUY, SELL, HOLD)

        Returns:
            Tuple of (observation, normalized_reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode already terminated. Call reset() first.")

        # --- Save previous state ---
        prev_cash = self.cash
        prev_holdings = self.holdings
        prev_price = self.current_price

        # --- Execute trade ---
        trade_value = self._execute_trade(action)
        self.last_trade_value = trade_value
        self.last_action = action

        # Track actions
        self.actions_history.append(action)
        if action != TradeAction.HOLD:
            self.trade_count += 1

        # Track buy actions after threshold step
        buy_step_limit = self.config.get("success_criteria", {}).get(
            "max_buys_after_step", {}
        )
        if isinstance(buy_step_limit, dict):
            limit_step = buy_step_limit.get("step", 999)
            if action == TradeAction.BUY and self.current_step > limit_step:
                self.buy_actions_after_step += 1

        # Track early sells
        sell_threshold = self.config.get("sell_threshold", 0)
        if action == TradeAction.SELL and self.current_step < sell_threshold:
            self.sell_before_threshold = True

        # --- Advance to next price ---
        self.current_step += 1
        if self.current_step < len(self.price_series):
            self.current_price = self.price_series[self.current_step]
        # else: keep last price (will terminate on max_steps)

        # --- Compute reward ---
        reward = compute_reward(
            prev_cash=prev_cash,
            prev_holdings=prev_holdings,
            prev_price=prev_price,
            curr_cash=self.cash,
            curr_holdings=self.holdings,
            curr_price=self.current_price,
            trade_value=trade_value,
            action=action,
            step=self.current_step - 1,  # step at which action was taken
            task_config=self.config,
        )

        # --- Track drawdown ---
        portfolio_value = self.cash + self.holdings * self.current_price
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

        # --- Check termination ---
        self.done = check_done(
            self.cash, self.holdings, self.current_price,
            self.current_step, self.config,
        )

        # --- Apply terminal bonus if done ---
        if self.done:
            success = success_criteria_met(
                final_portfolio_value=portfolio_value,
                trade_count=self.trade_count,
                max_drawdown=self.max_drawdown,
                buy_actions_after_step=self.buy_actions_after_step,
                final_holdings=self.holdings,
                sell_before_threshold=self.sell_before_threshold,
                task_config=self.config,
            )
            reward = apply_terminal_bonus(reward, success, self.config)

        self.rewards_history.append(reward.normalized_reward)

        # --- Build observation ---
        obs = self._build_observation()

        return obs, reward.normalized_reward, self.done, {}

    def state(self) -> dict:
        """Return current state snapshot — callable independently of step()."""
        return {
            "step": self.current_step,
            "price": self.current_price,
            "cash": self.cash,
            "holdings": self.holdings,
            "portfolio_value": self.cash + self.holdings * self.current_price,
            "scenario": self.scenario_label,
            "done": self.done,
        }

    def close(self):
        """Clean up resources."""
        pass  # No external resources to release

    def compute_final_score(self) -> float:
        """
        Compute the final grading score for the completed episode.
        Delegates to grader.grader.grade().
        """
        from grader.grader import grade

        trajectory = {
            "final_portfolio_value": self.cash + self.holdings * self.current_price,
            "trade_count": self.trade_count,
            "max_drawdown": self.max_drawdown,
            "buy_actions_after_step": self.buy_actions_after_step,
            "final_holdings": self.holdings,
            "sell_before_threshold": self.sell_before_threshold,
            "actions": [a.value for a in self.actions_history],
            "rewards": self.rewards_history,
        }
        return grade(self.config["task_name"], trajectory, self.config)

    # --- Private helpers ---

    def _execute_trade(self, action: TradeAction) -> float:
        """
        Execute a trade and return the absolute trade value.

        BUY:  spend all available cash to buy units at current price
        SELL: sell all holdings at current price
        HOLD: no trade

        Transaction costs are applied as a fraction of trade value.
        """
        cost_rate = self.config.get("transaction_cost_rate", 0.001)

        if action == TradeAction.BUY and self.cash > 0:
            # Buy as many units as cash allows (after costs)
            effective_cash = self.cash / (1 + cost_rate)
            units = effective_cash / self.current_price
            trade_value = units * self.current_price
            cost = trade_value * cost_rate
            self.holdings += units
            self.cash -= (trade_value + cost)
            # Fix floating point: ensure cash doesn't go negative
            if self.cash < 0.001:
                self.cash = 0.0
            return trade_value

        elif action == TradeAction.SELL and self.holdings > 0:
            trade_value = self.holdings * self.current_price
            cost = trade_value * cost_rate
            self.cash += (trade_value - cost)
            self.holdings = 0.0
            return trade_value

        # HOLD or no valid trade
        return 0.0

    def _build_observation(self) -> TradeObservation:
        """Build a TradeObservation from current state."""
        # Price history: up to last 5 prices
        start = max(0, self.current_step - 4)
        price_history = self.price_series[start: self.current_step + 1]

        # MA5: average of last 5 prices (or fewer if early in episode)
        if len(price_history) > 0:
            ma5 = sum(price_history) / len(price_history)
        else:
            ma5 = self.current_price

        # Average volatility: mean absolute pct change over available history
        if len(price_history) >= 2:
            changes = []
            for i in range(1, len(price_history)):
                if price_history[i - 1] > 0:
                    changes.append(abs(price_history[i] - price_history[i - 1]) / price_history[i - 1])
            avg_volatility = sum(changes) / len(changes) if changes else 0.0
        else:
            avg_volatility = 0.0

        portfolio_value = self.cash + self.holdings * self.current_price

        obs = TradeObservation(
            price_history=price_history,
            current_price=self.current_price,
            ma5=round(ma5, 4),
            avg_volatility=round(avg_volatility, 6),
            cash=round(self.cash, 2),
            holdings=round(self.holdings, 4),
            portfolio_value=round(portfolio_value, 2),
            step=self.current_step,
            max_steps=self.config["max_steps"],
            scenario_label=self.scenario_label,
            last_action=self.last_action,
            analyst_note="",  # placeholder, filled below
            done=self.done,
        )

        # Generate analyst note from current observation
        obs.analyst_note = generate_note(obs)

        return obs
