"""
Visualization utilities to plot QAD recovery benchmarks and learning curves.
"""

from typing import Dict, List, Optional
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_benchmark_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str = "benchmark_comparison.png",
    metric: str = "accuracy",
):
    """
    Plots a bar chart comparing models across configurations (Teacher, PTQ, QAT, QAD).

    Args:
        results: Dict mapping model variant name to metric dict {'accuracy': ..., 'ppl': ...}
        save_path: Output image filepath
        metric: 'accuracy' or 'ppl'
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    names = list(results.keys())
    values = [results[k][metric] for k in names]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Color palette
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    if len(colors) < len(names):
        colors = colors * (len(names) // len(colors) + 1)
    bar_colors = colors[: len(names)]

    bars = ax.bar(names, values, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.2)

    ylabel = "Top-1 Next-Token Accuracy (%)" if metric == "accuracy" else "Perplexity (Lower is better)"
    title = "Model Accuracy Comparison: Teacher vs PTQ vs QAT vs QAD" if metric == "accuracy" else "Perplexity Comparison"

    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        suffix = "%" if metric == "accuracy" else ""
        ax.annotate(
            f"{height:.2f}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Benchmark plot saved to: {save_path}")


def plot_learning_curves(
    curves: Dict[str, List[Dict[str, float]]],
    save_path: str = "learning_curves.png",
):
    """
    Plots training loss and validation accuracy curves comparing QAT vs QAD.

    Args:
        curves: Dict mapping run name (e.g., 'QAT', 'QAD') to list of epoch records.
        save_path: Output image filepath.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    styles = {
        "QAT": {"color": "#7570b3", "linestyle": "--", "marker": "o"},
        "QAD": {"color": "#1b9e77", "linestyle": "-", "marker": "s"},
    }

    for name, history in curves.items():
        epochs = [r["epoch"] for r in history]
        val_losses = [r["val_loss"] for r in history]
        val_accs = [r["val_accuracy"] for r in history]

        style = styles.get(name, {"color": "black", "linestyle": "-", "marker": "x"})

        ax1.plot(epochs, val_losses, label=name, **style, linewidth=2)
        ax2.plot(epochs, val_accs, label=name, **style, linewidth=2)

    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Validation Loss (Convergence)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Accuracy Recovery", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curves saved to: {save_path}")
