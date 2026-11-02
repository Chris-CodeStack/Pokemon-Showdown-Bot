import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def generate_calibration_plot(y_true_binary, y_prob):
    prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=8, strategy='uniform')

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfectly calibrated", alpha=0.7)

    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#1f77b4", label="Behavior Cloning Model")

    ax.set_title("Reliability Diagram (Calibration Curve)")
    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Actual Empirical Accuracy")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/calibration_curve.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("figures/calibration_curve.png", format="png", bbox_inches="tight")
    
    print("Success! Generated calibration curve saved to figures/calibration_curve.pdf")
    plt.show()

if __name__ == "__main__":
    dummy_binary = np.random.choice([0, 1], size=500, p=[0.3, 0.7])
    dummy_probs = np.random.uniform(0.4, 1.0, size=500)
    
    generate_calibration_plot(dummy_binary, dummy_probs)