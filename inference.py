"""
STRIP — LLM Inference Script

MANDATORY for submission. Uses the OpenAI client to run an LLM agent
through all tasks and emits structured stdout in the required format.

Environment variables:
    API_BASE_URL  — OpenAI-compatible API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — API key

Stdout format:
    [START] task=<task_name> env=strip model=<model_name>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import os
from openai import OpenAI
from env.environment import STRIPEnv
from env.models import TradeAction


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Minimum 3 required tasks for submission
TASKS = ["bullish", "bearish", "volatile", "sideways"]


def build_prompt(obs) -> str:
    """Build the LLM prompt from the current observation."""
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
    """Parse LLM response into a TradeAction, defaulting to HOLD."""
    text = text.strip().upper()
    for action in TradeAction:
        if action.value in text:
            return action
    return TradeAction.HOLD  # default fallback


def run_task(task_name: str):
    """Run a single task episode using the LLM agent."""
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
            f"[STEP] step={step + 1} action=HOLD "
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
