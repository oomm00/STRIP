"""
STRIP — FastAPI Server

Wraps the STRIPEnv environment as a REST API with OpenEnv-compliant endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from env.environment import STRIPEnv
from env.models import TradeAction


app = FastAPI(
    title="STRIP",
    description=(
        "Systematic Trading Reasoning Intelligence Platform — "
        "An offline financial advisory decision environment."
    ),
    version="1.0.0",
)

# Global environment instance
env = STRIPEnv()


# --- Request / Response schemas ---

class ResetRequest(BaseModel):
    task: str = "bullish"


class StepRequest(BaseModel):
    action: str  # "BUY" | "SELL" | "HOLD"


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    step: int
    price: float
    cash: float
    holdings: float
    portfolio_value: float
    scenario: str
    done: bool


# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "STRIP API is running. Go to /docs"}


@app.get("/health")
def health():
    """Health check — must return 200."""
    return {"status": "ok", "service": "strip"}


@app.post("/reset")
def reset(request: ResetRequest = None):
    """Start new episode, returns initial TradeObservation."""
    task = "bullish"
    if request and hasattr(request, "task"):
        task = request.task

    try:
        obs = env.reset(task=task)
        return obs.model_dump()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    """Submit TradeAction, returns (TradeObservation, reward, done, info)."""
    try:
        action = TradeAction(request.action.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {request.action}. Must be BUY, SELL, or HOLD.",
        )

    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=round(reward, 4),
            done=done,
            info=info,
        ).model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """Returns current state dict without advancing the episode."""
    return env.state()
