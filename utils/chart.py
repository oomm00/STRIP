"""
STRIP — Chart Utility

Generates a price + action chart for demo/analysis output.
Plots the price series with annotated BUY/SELL/HOLD markers.
"""

import os

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_episode(
    prices: list,
    actions: list,
    task_name: str,
    output_path: str = None,
) -> str:
    """
    Plot price series with action markers.

    Args:
        prices:     List of prices at each step (including step 0).
        actions:    List of actions taken (one per step after step 0).
        task_name:  Name of the task for the chart title.
        output_path: Where to save the chart. Defaults to 'charts/<task_name>.png'.

    Returns:
        Path to the saved chart image.
    """
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not installed — skipping chart generation.")
        return ""

    if output_path is None:
        os.makedirs("charts", exist_ok=True)
        output_path = os.path.join("charts", f"{task_name}.png")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot price line
    steps = list(range(len(prices)))
    ax.plot(steps, prices, color="#3b82f6", linewidth=2, label="Price", zorder=2)

    # Action markers
    action_colors = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#a3a3a3"}
    action_markers = {"BUY": "^", "SELL": "v", "HOLD": "o"}

    for i, action in enumerate(actions):
        step_idx = i + 1  # actions start at step 1
        if step_idx < len(prices):
            ax.scatter(
                step_idx,
                prices[step_idx],
                color=action_colors.get(action, "#a3a3a3"),
                marker=action_markers.get(action, "o"),
                s=100 if action != "HOLD" else 30,
                zorder=3,
                edgecolors="white",
                linewidth=0.5,
            )

    # Styling
    ax.set_title(f"STRIP — {task_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#22c55e", markersize=10, label="BUY"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#ef4444", markersize=10, label="SELL"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#a3a3a3", markersize=6, label="HOLD"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", facecolor="#1a1a2e", edgecolor="#333",
              labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[CHART] Saved to {output_path}")
    return output_path
