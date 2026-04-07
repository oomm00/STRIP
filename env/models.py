"""
STRIP — Data Models

Defines the core typed models for the STRIP environment:
- TradeAction: BUY / SELL / HOLD action enum
- TradeObservation: Full observation space (Pydantic)
- TradeReward: Decomposed reward components (Pydantic)
"""

from enum import Enum
from typing import List
from pydantic import BaseModel


class TradeAction(str, Enum):
    """Valid portfolio actions. Using str, Enum for JSON serialization."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeObservation(BaseModel):
    """
    Full observation delivered to the agent at each step.

    Contains market state, portfolio state, episode metadata,
    analyst reasoning, and termination flag.
    """

    # Market state
    price_history: List[float]   # closing prices for last N steps
    current_price: float         # price at current timestep
    ma5: float                   # 5-step moving average
    avg_volatility: float        # mean absolute price change over last 5 steps

    # Portfolio state
    cash: float                  # available cash
    holdings: float              # units currently held
    portfolio_value: float       # cash + holdings * current_price

    # Episode metadata
    step: int                    # current timestep (0-indexed)
    max_steps: int               # total episode length for this task
    scenario_label: str          # "bullish" | "bearish" | "sideways" | "volatile" etc.
    last_action: TradeAction     # action taken at previous step

    # Analyst reasoning (always present, never optional)
    analyst_note: str            # deterministic structured reasoning note

    # Termination
    done: bool                   # True when episode has ended


class TradeReward(BaseModel):
    """
    Decomposed reward returned at each step.

    All graders use normalized_reward (in [0.0, 1.0]).
    The raw components are exposed for debugging and analysis.
    """
    raw_reward: float           # unscaled step reward
    normalized_reward: float    # reward in [0.0, 1.0] — used by all graders
    delta_v: float              # portfolio value change this step
    transaction_cost: float     # C_t component
    position_risk: float        # risk_t component
    early_exit_penalty: float   # E_t component
    terminal_bonus: float       # +10 or -10 at episode end, else 0
