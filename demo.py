"""
STRIP — Offline Demo

Runs one complete episode using the rule-based agent (no LLM required).
Prints step-by-step output showing price, portfolio state, analyst note, and action.

Usage:
    python demo.py --task bullish
    python demo.py --task bearish_bounce
"""

import argparse

from env.environment import STRIPEnv
from env.models import TradeAction
from agents.trader import choose_action


def run_demo(task_name: str):
    """Run a single episode with the rule-based trader and print each step."""
    env = STRIPEnv()
    obs = env.reset(task=task_name)

    initial_value = obs.portfolio_value
    print(f"\n{'='*70}")
    print(f"[DEMO] Task: {task_name} | Initial capital: {initial_value:.0f}")
    print(f"{'='*70}\n")

    step = 0
    done = False
    rewards = []

    while not done:
        action = choose_action(obs)
        obs, reward, done = env.step(action)
        rewards.append(reward)
        step += 1

        print(
            f"Step {step:2d} | "
            f"Price: {obs.current_price:8.2f} | "
            f"Cash: {obs.cash:10.2f} | "
            f"Holdings: {obs.holdings:8.2f} | "
            f"Portfolio: {obs.portfolio_value:10.2f}"
        )
        print(f"  Analyst: \"{obs.analyst_note}\"")
        print(
            f"  Action:  {action.value:4s} | "
            f"Reward: {reward:.4f} | "
            f"Done: {str(done).lower()}"
        )
        print()

    # Final summary
    score = env.compute_final_score()
    final_value = obs.portfolio_value
    success = score >= 0.6

    print(f"{'='*70}")
    print(
        f"[DEMO END] Final portfolio: {final_value:.2f} | "
        f"Score: {score:.2f} | "
        f"Success: {str(success).lower()}"
    )
    print(f"  Total steps: {step} | Total trades: {env.trade_count}")
    print(f"  Max drawdown: {env.max_drawdown:.2%}")
    reward_str = ", ".join(f"{r:.4f}" for r in rewards)
    print(f"  Rewards: [{reward_str}]")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="STRIP — Offline Demo")
    parser.add_argument(
        "--task",
        type=str,
        default="bullish",
        help="Task name (e.g. bullish, bearish, volatile, sideways, bullish_trap, etc.)",
    )
    args = parser.parse_args()
    run_demo(args.task)


if __name__ == "__main__":
    main()
