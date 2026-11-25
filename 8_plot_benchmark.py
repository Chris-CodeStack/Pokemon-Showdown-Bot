import os
import numpy as np
import matplotlib.pyplot as plt

# Academic styling matching your other plots
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
})

def generate_benchmark_plot():
    opponents = ["Random Baseline\n(Avg 26.0 turns)", "Greedy Damage Baseline\n(Avg 21.0 turns)"]
    win_rates = [55.0, 15.0]
    records = ["11/20", "3/20"]
    colors = ["#2ca02c", "#d62728"]

    x = np.arange(len(opponents))
    width = 0.45

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

    bars = ax.bar(
        x,
        win_rates,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    # 50% break-even reference line
    ax.axhline(50.0, linestyle="--", color="black", linewidth=1.2, alpha=0.75, label="50% Break-Even")

    # Annotate each bar with win rate and match record
    for bar, rec in zip(bars, records):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}% ({rec})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_title("Live Battle Win Rate (Gen 9 Random Doubles, 20 Matches Each)")
    ax.set_ylabel("BC Bot Win Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(opponents)
    ax.set_ylim(0, 72)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/live_benchmark.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("figures/live_benchmark.png", format="png", bbox_inches="tight")
    print("Saved benchmark chart to figures/live_benchmark.pdf and figures/live_benchmark.png")
    plt.show()

if __name__ == "__main__":
    generate_benchmark_plot()