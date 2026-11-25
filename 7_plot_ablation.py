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

def generate_ablation_plot():
    # Update these if your final test-set numbers differ slightly
    metrics = ["Top-1 Accuracy", "Top-3 Accuracy"]
    baseline_5feat = [14.0, 22.0]     # Initial 5-dim state (4 HPs + turn)
    enriched_132feat = [41.0, 74.0]   # Enriched 132-dim state (perspective + alive + species)

    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

    bars1 = ax.bar(
        x - width / 2,
        baseline_5feat,
        width,
        label="Baseline (5 Features)",
        color="#7f8c8d",
        edgecolor="black",
        linewidth=0.6,
    )
    bars2 = ax.bar(
        x + width / 2,
        enriched_132feat,
        width,
        label="Enriched State (132 Features)",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
    )

    # Add percentage labels directly above each bar
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_title("Action Prediction Accuracy")
    ax.set_ylabel("Validation / Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 88)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/feature_ablation.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("figures/feature_ablation.png", format="png", bbox_inches="tight")
    print("Saved ablation chart to figures/feature_ablation.pdf and figures/feature_ablation.png")
    plt.show()

if __name__ == "__main__":
    generate_ablation_plot()