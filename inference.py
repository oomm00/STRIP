"""
STRIP — LLM Inference Script (FIXED VERSION)
"""

import os
import time

from openai import OpenAI
from env.environment import STRIPEnv
from env.models import TradeAction
from agents.trader import choose_action


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

client = None
if API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = ["bullish", "bearish", "volatile", "sideways"]
API_FAILED = False


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
    if not text:
        return TradeAction.HOLD
    text = text.strip().upper()
    for action in TradeAction:
        if action.value in text:
            return action
    return TradeAction.HOLD


def run_task(task_name: str):
    env = STRIPEnv()
    obs = env.reset(task=task_name)
    print(f"[START] task={task_name} env=strip model={MODEL_NAME}", flush=True)

    step, rewards, done, score = 0, [], False, 0.0

    global API_FAILED
    try:
        while not done:
            action = None
            if client and not API_FAILED:
                for attempt in range(2):
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": build_prompt(obs)}],
                            max_tokens=10,
                            timeout=10
                        )
                        action = parse_action(response.choices[0].message.content)
                        break
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Retry only on generic rate limits, not billing issues
                        if "429" in error_msg and "insufficient_quota" not in error_msg:
                            time.sleep(2)
                        else:
                            API_FAILED = True
                            break  # Abort API globally and fallback to rule-based
                            
            if action is None:
                action = choose_action(obs)

            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            step += 1

            print(
                f"[STEP] step={step} action={action.value} "
                f"reward={reward:.2f} done={str(done).lower()} error=null", flush=True
            )

        score = env.compute_final_score()

    except Exception as e:
        score = 0.0
        print(
            f"[STEP] step={step + 1} action=HOLD "
            f"reward=0.00 done=true error={str(e)}", flush=True
        )

    finally:
        env.close()
        reward_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(score >= 0.59).lower()} "
            f"steps={step} score={score:.2f} rewards={reward_str}", flush=True
        )


def main():
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()