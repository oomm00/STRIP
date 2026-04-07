---
title: STRIP
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
license: mit
short_description: 'short_description: AI agent for financial trading decisions'
---
# STRIP — Systematic Trading Reasoning Intelligence Platform

## Live Demo

- API Docs: https://oomm00-strip.hf.space/docs
- Health Check: https://oomm00-strip.hf.space/health

> An offline, OpenEnv-compliant environment where an AI agent acts as a financial advisor:
> reading a market brief, interpreting an analyst's reasoning, and recommending portfolio actions
> (BUY / SELL / HOLD) across deterministic market scenarios — exactly the decision loop a human
> advisor executes daily.

---

## Table of Contents

1. [Environment Description and Motivation](#1-environment-description-and-motivation)
2. [Action and Observation Spaces](#2-action-and-observation-spaces)
3. [Reward Function](#3-reward-function)
4. [Tasks and Graders](#4-tasks-and-graders)
5. [Termination Conditions](#5-termination-conditions)
6. [Agent Architecture](#6-agent-architecture)
7. [OpenEnv Spec Compliance](#7-openenv-spec-compliance)
8. [Repository Structure](#8-repository-structure)
9. [Setup and Usage](#9-setup-and-usage)
10. [Deployment](#10-deployment)
11. [Baseline Scores](#11-baseline-scores)
12. [Pre-Submission Checklist](#12-pre-submission-checklist)

---

## 1. Environment Description and Motivation

### What This Models

Financial advisors do not trade by staring at a price chart. They read a market brief prepared
by an analyst, interpret the reasoning, cross-reference their client's current portfolio exposure,
and recommend an action. That three-step loop — **brief → reasoning → recommendation** — is the
real-world task STRIP simulates.

The agent plays the role of the advisor. At each step it receives:
- a structured market observation (prices, portfolio state, scenario context)
- a deterministic analyst note synthesizing that state into plain-English reasoning

It then recommends BUY, SELL, or HOLD. The environment scores the recommendation against
deterministic success criteria defined per task.

### Why This Is a Real-World Task (Not a Game)

Portfolio action recommendation is a task with:
- real economic consequences
- a defined reasoning process (trend analysis → risk assessment → action)
- clear success metrics (capital preservation, growth targets, drawdown limits)
- a natural fit for LLM agents — the agent reads language and produces a decision

This is distinct from toy trading bots that predict prices. The agent is evaluated on its
**decision quality given a brief**, not on market prediction.

---

## 2. Action and Observation Spaces

### Action Space (`env/models.py`)

```python
from enum import Enum

class TradeAction(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
```

Using `Enum` enforces type safety — the environment rejects any action not in this set.

---

### Observation Space (`env/models.py`)

```python
from pydantic import BaseModel
from typing import List

class TradeObservation(BaseModel):
    # Market state
    price_history:    List[float]  # closing prices for last N steps
    current_price:    float        # price at current timestep
    ma5:              float        # 5-step moving average
    avg_volatility:   float        # mean absolute price change over last 5 steps

    # Portfolio state
    cash:             float        # available cash
    holdings:         float        # units currently held
    portfolio_value:  float        # cash + holdings * current_price

    # Episode metadata
    step:             int          # current timestep (0-indexed)
    max_steps:        int          # total episode length for this task
    scenario_label:   str          # "bullish" | "bearish" | "sideways" | "volatile"
    last_action:      TradeAction  # action taken at previous step

    # Analyst reasoning (always present, never optional)
    analyst_note:     str          # deterministic structured reasoning note

    # Termination
    done:             bool         # True when episode has ended
```

All fields use Pydantic for runtime type validation. The `analyst_note` field is non-optional —
every observation carries an inspectable reasoning trace so the grader can always see the
agent's decision context.

---

### Reward Model (`env/models.py`)

```python
class TradeReward(BaseModel):
    raw_reward:         float  # unscaled step reward
    normalized_reward:  float  # reward in [0.0, 1.0] — used by all graders
    delta_v:            float  # portfolio value change this step
    transaction_cost:   float  # C_t component
    position_risk:      float  # risk_t component
    early_exit_penalty: float  # E_t component
    terminal_bonus:     float  # +10 or -10 at episode end, else 0
```

---

## 3. Reward Function

### Formula

```
R_t = delta_V_t  -  1.0 * C_t  -  0.5 * risk_t  -  2.0 * E_t
```

Every term defined explicitly — no ambiguity:

| Term | Formula | What It Penalises |
|------|---------|-------------------|
| `delta_V_t` | `V_t - V_{t-1}` where `V = cash + holdings * price` | Core profit/loss signal |
| `C_t` | `|trade_value| * 0.001` | Overtrading (0.1% per trade) |
| `risk_t` | `holdings_t * |price_t - price_{t-1}| / price_{t-1}` | Holding large positions during volatile moves (agent-controllable) |
| `E_t` | `1 if SELL and step < task_config["sell_threshold"] else 0` | Premature exits in trend tasks |

Constants: `alpha = 1.0`, `beta = 0.5`, `gamma = 2.0` — fixed across all tasks.

### Why `risk_t` Not Raw Volatility

Penalising raw market volatility punishes the agent for something it cannot control. `risk_t`
penalises the **choice** to hold a large position when the market moves fast — a decision the
agent makes. This produces a cleaner, fairer learning signal.

### Normalization to [0.0, 1.0]

All graders receive normalized reward. Normalization uses a **per-step scale** of 5% of
initial capital to ensure step-level reward variations are meaningful (not crushed near 0.5):

```python
def normalize_reward(raw_reward: float, initial_capital: float) -> float:
    scale = initial_capital * 0.05  # per-step scale = 5% of capital
    clipped = max(-scale, min(scale, raw_reward))
    return (clipped + scale) / (2 * scale)
```

This maps:
- `-scale` → `0.0`
- `0` → `0.5`
- `+scale` → `1.0`

Values beyond `±scale` are clipped. The 5% scale ensures that typical step-level profit/loss
signals (a few dollars on ~10,000 capital) produce rewards spread meaningfully across [0, 1],
not clustered at 0.5.

### Partial Progress Signals

The reward provides signal at **every step**, not just at episode end:
- Positive `delta_V` rewards correct positioning immediately
- `C_t` penalises each unnecessary trade as it happens
- `risk_t` penalises large risky positions in real time
- Terminal bonus of `+10` / `-10` aligns the trajectory with task success

The reward is never sparse.

### Terminal Bonus

```python
if success_criteria_met(final_state, task_config):
    terminal_reward = +10.0
else:
    terminal_reward = -10.0
```

### Full Implementation (`env/reward.py`)

```python
def compute_reward(prev_state, curr_state, action, step, task_config):
    V_prev = prev_state.cash + prev_state.holdings * prev_state.current_price
    V_curr = curr_state.cash + curr_state.holdings * curr_state.current_price
    delta_v = V_curr - V_prev

    trade_value = abs(curr_state.last_trade_value)
    C_t = trade_value * 0.001

    sigma_t = abs(curr_state.current_price - prev_state.current_price) / prev_state.current_price
    risk_t = curr_state.holdings * sigma_t

    E_t = 1.0 if (action == TradeAction.SELL and step < task_config["sell_threshold"]) else 0.0

    raw = delta_v - 1.0 * C_t - 0.5 * risk_t - 2.0 * E_t
    normalized = normalize_reward(raw, task_config["initial_capital"])
    return raw, normalized
```

---

## 4. Tasks and Graders

All task data lives in `tasks/*.json`. Each file contains:
- pre-generated hardcoded price series (identical across all machines — no runtime RNG or formula)
- initial state (capital, holdings)
- binary-checkable success thresholds
- edge case variant (in separate JSON files)

All price series are fully hardcoded in `tasks/*.json`. No random number generation occurs
during episodes. Running the same grader twice on the same trajectory always returns the
same score.

Graders in `grader/grader.py` each return a float in `[0.0, 1.0]`.

---

### Task 1 — Bullish Market (Easy)

**Scenario:** Prices rise steadily ~1.5% per step for 20 steps.

**Initial state:** Cash 10,000 | Holdings 0 | Transaction cost 0.1%

**Expected behavior:** Enter position early (BUY within first 5 steps), HOLD through the trend,
avoid SELL before step 10.

**Success criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Final portfolio value | >= 115% of initial (>= 11,500) |
| Total trades | <= 5 |
| SELL before step 10 | Not allowed |

**Grader:**
```python
score = 0.0
if final_value >= 11500:   score += 0.5
if trade_count <= 5:       score += 0.3
if no_early_sell:          score += 0.2
```

**Edge case — Bull Trap (`bullish_trap.json`):** Uptrend until step 12, then -10% drop at step 13.
Success threshold drops to 108%. Tests overconfidence and recovery.

---

### Task 2 — Bearish Market (Medium)

**Scenario:** Prices fall steadily ~1.2% per step for 20 steps.

**Initial state:** Cash 5,000 | Holdings 500 units | Transaction cost 0.1%

**Expected behavior:** Reduce holdings early, avoid BUY, preserve capital above 95%.

**Success criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Final portfolio value | >= 95% of initial (>= 9,025) |
| BUY actions after step 5 | 0 |
| Holdings at end | <= 100 units |

**Grader:**
```python
score = 0.0
if final_value >= 9025:       score += 0.5
if no_buy_after_step5:        score += 0.3
if final_holdings <= 100:     score += 0.2
```

**Edge case — Dead Cat Bounce (`bearish_bounce.json`):** Temporary +5% spike at step 8 mid-downtrend.
Tests whether the agent incorrectly re-enters on a false reversal signal.

---

### Task 3 — Volatile Market (Hard)

**Scenario:** Prices swing +-5 to 8% per step for 20 steps.

**Initial state:** Cash 5,000 | Holdings 200 units | Transaction cost 0.1%

**Expected behavior:** Reduce position size to manage `risk_t`, avoid impulsive BUY during
spikes, keep drawdown under 15%.

**Success criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Final portfolio value | >= 90% of initial (>= 9,000) |
| Max single-step drawdown | <= 15% |
| Total trades | <= 6 |

**Grader:**
```python
score = 0.0
if final_value >= 9000:        score += 0.4
if max_drawdown <= 0.15:       score += 0.4
if trade_count <= 6:           score += 0.2
```

**Edge case — Flash Crash (`volatile_crash.json`):** Single -20% price drop at step 7. Tests the capital-collapse
termination condition and clean episode handling.

**Why this is hard:** The agent must balance `risk_t` reduction (SELL) against `C_t`
accumulation (each trade costs 0.1%). Optimal play requires anticipating volatility, not
reacting to it.

---

### Task 4 — Sideways Market (Easy-Medium)

**Scenario:** Prices oscillate +-1% around a base price for 20 steps.

**Initial state:** Cash 10,000 | Holdings 0 | Transaction cost 0.1%

**Expected behavior:** HOLD most steps; overtrading destroys capital in flat markets.

**Success criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Final portfolio value | >= 98% of initial (>= 9,800) |
| Total trades | <= 3 |

**Grader:**
```python
score = 0.0
if final_value >= 9800:   score += 0.6
if trade_count <= 3:      score += 0.4
```

**Edge case — Whipsaw (`sideways_whipsaw.json`):** Oscillations increase to +-3% in steps 10-15. `risk_t` spikes for
agents holding large positions. Tests whether the agent adjusts exposure to noise.

---

## 5. Termination Conditions

```python
def check_done(state, step, config) -> bool:
    V_t = state.cash + state.holdings * state.current_price

    # 1. Max steps reached
    if step >= config["max_steps"]:
        return True

    # 2. Capital collapse: portfolio below 80% of initial
    if V_t <= 0.8 * config["initial_capital"]:
        return True

    # 3. No active position (cash = 0 AND holdings = 0)
    if state.cash == 0 and state.holdings == 0:
        return True

    return False
```

Thresholds load from `task_config` (not hardcoded per scenario). Terminal reward applies
on the step where `done` becomes True.

---

## 6. Agent Architecture

### Agent A — Analyst (`agents/analyst.py`)

Generates a **deterministic structured note** from the current market state. No LLM, no
randomness — identical state always produces identical note. The note is embedded in every
`TradeObservation` before the Trader sees it.

```python
def generate_note(state: TradeObservation) -> str:
    trend = "upward" if state.current_price > state.ma5 else "downward"
    vol_label = "high" if state.avg_volatility > 0.03 else "low"
    vol_pct = state.avg_volatility * 100
    exposure = (state.holdings * state.current_price / state.portfolio_value
                if state.portfolio_value > 0 else 0)

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
```

### Agent B — Trader (`agents/trader.py`)

Rule-based offline baseline that reads the `analyst_note` to choose an action. In `inference.py`
this logic is replaced by an LLM call via the OpenAI client.

```python
def choose_action(obs: TradeObservation) -> TradeAction:
    note = obs.analyst_note.lower()

    if "reduc" in note:
        if obs.holdings > 0:
            return TradeAction.SELL
        return TradeAction.HOLD
    elif "momentum" in note:
        if obs.holdings == 0 and obs.cash > 0:
            return TradeAction.BUY
        return TradeAction.HOLD

    return TradeAction.HOLD
```

---

## 7. OpenEnv Spec Compliance

### Interface Methods (`env/environment.py`)

```python
class STRIPEnv:

    def reset(self, task: str = "bullish") -> TradeObservation:
        """Load task config, initialise state, return first observation."""

    def step(self, action: TradeAction) -> tuple[TradeObservation, float, bool]:
        """
        1. Update market state (advance price, update cash/holdings)
        2. Analyst generates note from new state
        3. Build TradeObservation (note included)
        4. Compute reward (raw + normalized)
        5. Check termination
        Returns: (observation, normalized_reward, done)
        """

    def state(self) -> dict:
        """Return current state snapshot — callable independently of step()."""
        return {
            "step":            self.current_step,
            "price":           self.current_price,
            "cash":            self.cash,
            "holdings":        self.holdings,
            "portfolio_value": self.cash + self.holdings * self.current_price,
            "scenario":        self.scenario_label,
            "done":            self.done,
        }

    def close(self):
        """Clean up resources."""
```

### openenv.yaml (repo root)

```yaml
name: strip
version: "1.0.0"
description: >
  Offline financial advisory decision environment. An AI agent reads a structured
  market brief and analyst note, then recommends portfolio actions across
  deterministic market scenarios.
tasks:
  - name: bullish
    difficulty: easy
  - name: bullish_trap
    difficulty: easy_hard
  - name: bearish
    difficulty: medium
  - name: bearish_bounce
    difficulty: medium_hard
  - name: volatile
    difficulty: hard
  - name: volatile_crash
    difficulty: hard
  - name: sideways
    difficulty: easy_medium
  - name: sideways_whipsaw
    difficulty: medium
observation_type: TradeObservation
action_type: TradeAction
state_type: dict
reward_range: [0.0, 1.0]
episode_length: 20
offline: true
```

### Validation

```bash
openenv validate .
```

Expected: all checks pass — typed models, `step()`, `reset()`, `state()`, `openenv.yaml` present.

---

## 8. Repository Structure

```
STRIP/
|
|- inference.py              # MANDATORY: LLM baseline runner (OpenAI client, stdout format)
|- openenv.yaml              # OpenEnv metadata and spec
|- app.py                    # FastAPI server wrapping the environment
|- demo.py                   # Offline demo — runs one episode, prints each step
|- requirements.txt
|- Dockerfile
|- README.md
|
|- env/
|   |- __init__.py
|   |- environment.py        # STRIPEnv: reset(), step(), state(), close()
|   |- models.py             # TradeObservation, TradeAction (Enum), TradeReward (Pydantic)
|   +- reward.py             # compute_reward(), normalize_reward(), success_criteria_met()
|
|- agents/
|   |- __init__.py
|   |- analyst.py            # generate_note() — deterministic template-driven
|   +- trader.py             # choose_action() — rule-based baseline policy
|
|- grader/
|   |- __init__.py
|   +- grader.py             # per-task graders, all return float in [0.0, 1.0]
|
|- tasks/
|   |- bullish.json          # hardcoded price series + config + success thresholds
|   |- bullish_trap.json     # edge case: uptrend then -10% drop
|   |- bearish.json
|   |- bearish_bounce.json   # edge case: +5% dead cat bounce at step 8
|   |- volatile.json
|   |- volatile_crash.json   # edge case: -20% flash crash at step 7
|   |- sideways.json
|   +- sideways_whipsaw.json # edge case: oscillations increase to +-3%
|
|- data/
|   |- stock.csv             # base price reference data
|   +- scenario_context.json # offline scenario descriptors (no external APIs)
|
+- utils/
    |- __init__.py
    +- chart.py              # optional: price + action chart for demo output
```

**Note on `scenario_context.json`:** Contains predefined offline descriptors. No news data,
no external APIs. Example:

```json
{
  "scenario": "bullish",
  "trend": "upward",
  "volatility": "low",
  "description": "steady growth with minimal price fluctuation"
}
```

Does not contain `"optimal_action"` — purely descriptive, not prescriptive.

---

## 9. Setup and Usage

### Prerequisites

```
Python 3.10+
Docker
```

### Local Development (Uvicorn)

```bash
git clone https://github.com/<your-hf-username>/strip
cd strip
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Available endpoints:
- `GET  /health`  — server status (must return 200)
- `GET  /docs`    — auto-generated OpenAPI documentation
- `POST /reset`   — start new episode, returns initial TradeObservation
- `POST /step`    — submit TradeAction, returns (TradeObservation, reward, done)
- `GET  /state`   — returns current state dict without advancing the episode

### Docker

```bash
docker build -t strip .
docker run -p 8000:8000 strip
```

### Offline Demo (no server required)

```bash
python demo.py --task bullish
```

Example output:

```
[DEMO] Task: bullish | Initial capital: 10000

Step  1 | Price: 101.50 | Cash: 0.00 | Holdings: 98.42 | Portfolio: 10139.61
Analyst: "Price trend is upward over last 5 steps (MA5: 100.75 -> current: 101.50).
          Volatility is low (1.5% avg move). Exposure is 100%. Bias: hold and monitor."
Action:  BUY | Reward: 0.5064 | Done: false

...

[DEMO END] Final portfolio: 13256.69 | Score: 1.00 | Success: true
```

### Running the LLM Inference Script

Set environment variables:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
```

Run:

```bash
python inference.py
```

---

## Inference Script Specification (`inference.py`)

Mandatory for submission. Rules:
- Named exactly `inference.py`, placed at repo root
- Uses the **OpenAI client** for all LLM calls (not raw HTTP)
- Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment variables (never hardcoded)
- Emits structured stdout in exactly this format — any deviation causes scoring failure

### Required Stdout Format

```
[START] task=<task_name> env=strip model=<model_name>
[STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

Rules:
- One `[START]` at episode begin
- One `[STEP]` per step, immediately after `env.step()` returns
- One `[END]` after `env.close()`, always emitted including on exception
- `reward` and `rewards` to 2 decimal places
- `done` and `success` lowercase: `true` or `false`
- `error` is the error string or `null`
- All fields on a single line — no newlines within a line
- Score must be in `[0.0, 1.0]`

### Example Output

```
[START] task=bullish env=strip model=gpt-4o-mini
[STEP] step=1 action=BUY reward=0.54 done=false error=null
[STEP] step=2 action=HOLD reward=0.52 done=false error=null
[STEP] step=3 action=HOLD reward=0.51 done=false error=null
[END] success=true steps=20 score=0.87 rewards=0.54,0.52,0.51,...
```

### Full Script (`inference.py`)

```python
import os
from openai import OpenAI
from env.environment import STRIPEnv
from env.models import TradeAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
TASKS  = ["bullish", "bearish", "volatile", "sideways"]

def build_prompt(obs) -> str:
    return (
        f"You are a financial advisor. Here is the current market state:\n\n"
        f"Scenario: {obs.scenario_label}\n"
        f"Current price: {obs.current_price:.2f}\n"
        f"Portfolio value: {obs.portfolio_value:.2f}\n"
        f"Cash: {obs.cash:.2f} | Holdings: {obs.holdings:.2f}\n"
        f"Step: {obs.step}/{obs.max_steps}\n\n"
        f"Analyst note: {obs.analyst_note}\n\n"
        f"Choose one action: BUY, SELL, or HOLD. Reply with only the action word."
    )

def parse_action(text: str) -> TradeAction:
    text = text.strip().upper()
    for action in TradeAction:
        if action.value in text:
            return action
    return TradeAction.HOLD  # default fallback

def run_task(task_name: str):
    env = STRIPEnv()
    obs = env.reset(task=task_name)
    print(f"[START] task={task_name} env=strip model={MODEL_NAME}")

    step, rewards, done, score = 0, [], False, 0.0

    try:
        while not done:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": build_prompt(obs)}],
                max_tokens=10,
            )
            action = parse_action(response.choices[0].message.content)
            obs, reward, done = env.step(action)
            rewards.append(reward)
            step += 1
            print(
                f"[STEP] step={step} action={action.value} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

        score = env.compute_final_score()

    except Exception as e:
        score = 0.0
        print(
            f"[STEP] step={step+1} action=HOLD "
            f"reward=0.00 done=true error={str(e)}"
        )

    finally:
        env.close()
        reward_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(score >= 0.6).lower()} "
            f"steps={step} score={score:.2f} rewards={reward_str}"
        )

def main():
    for task in TASKS:
        run_task(task)

if __name__ == "__main__":
    main()
```

---

## 10. Deployment

### Hugging Face Spaces

> Replace `<your-hf-username>` with your actual Hugging Face username before submitting.

```bash
openenv push --repo-id <your-hf-username>/strip
```

Provides:
- Live server at `https://<your-hf-username>-strip.hf.space` (must return 200 on `GET /health`)
- Pip-installable client: `pip install git+https://huggingface.co/spaces/<your-hf-username>/strip`
- Docker image: `docker pull registry.hf.space/<your-hf-username>-strip:latest`

### Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Infrastructure Constraints

| Constraint | Value |
|-----------|-------|
| Inference runtime | < 20 minutes total |
| Hardware target | vCPU=2, memory=8GB |
| Network | Fully offline during episodes |

### Scaling Config

| Variable | Default | Effect |
|----------|---------|--------|
| `WORKERS` | 2 | Parallel processes |
| `MAX_CONCURRENT_ENVS` | 10 | Simultaneous agent connections |
| `PORT` | 8000 | Listening port |

---

## 11. Baseline Scores

Scores produced by running `demo.py` with the rule-based agent against all tasks.
All price series are hardcoded — scores are fully reproducible across machines.

| Task | Difficulty | Score | Steps | Trades |
|------|-----------|-------|-------|--------|
| Bullish | Easy | 1.00 | 20 | 1 |
| Bearish | Medium | 1.00 | 20 | 1 |
| Volatile | Hard | 1.00 | 20 | 1 |
| Sideways | Easy-Med | 0.00 | 20 | 19 |
| Bullish Trap | Easy-Hard | 0.50 | 20 | 3 |
| Bearish Bounce | Medium-Hard | 0.70 | 20 | 3 |
| Volatile Crash | Hard | 1.00 | 20 | 1 |
| Sideways Whipsaw | Medium | 0.00 | 20 | 16 |

To reproduce:

```bash
python demo.py --task bullish
python demo.py --task bearish
python demo.py --task volatile
python demo.py --task sideways
```

---

## 12. Pre-Submission Checklist

| Item | Done |
|------|------|
| `openenv validate .` passes | - |
| `docker build && docker run` succeeds | - |
| HF Space `/health` returns 200 | - |
| HF Space responds to `reset()` | - |
| `inference.py` runs all tasks without error | - |
| `inference.py` stdout matches `[START]`/`[STEP]`/`[END]` format exactly | - |
| All graders return scores in `[0.0, 1.0]` | - |
| Reward is non-sparse (signal at every step) | - |
| All 3 termination conditions implemented and tested | - |
| `openenv.yaml` present at repo root | - |
| `state()` implemented and returns current state dict | - |
| README documents action space, observation space, setup instructions | - |
| Baseline scores documented and reproducible via `demo.py` | - |
| No external API calls during episode execution | - |
| Inference runtime under 20 minutes | - |
| No `TradeMind` references anywhere in codebase | - |
| `env=strip` in inference stdout (not `env=trademind`) | - |

---

*STRIP simulates the financial advisory decision loop: read a brief, interpret reasoning,
recommend an action. Built around the OpenEnv principle that environments should be isolated,
typed, deterministic, and deployable.*
