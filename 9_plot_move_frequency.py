import os
import json
from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
from bot import VGCBehaviorCloningNet

# Academic styling matching your other plots
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
})


def generate_move_frequency_plot():
    # 1. Load dataset and vocabulary
    data = np.load("data/processed_dataset.npz")
    X, y = data["X"], data["y"]

    with open("data/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    id_to_move = {int(k): v for k, v in vocab["id_to_move"].items()}
    state_dim = vocab["state_dim"]
    num_actions = vocab["num_actions"]

    # 2. Reproduce the exact 15% test split from 3_train_bc_model.py
    np.random.seed(42)
    indices = np.random.permutation(len(y))
    val_end = int(0.85 * len(y))
    test_idx = indices[val_end:]

    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = y[test_idx]

    # 3. Load trained model and predict on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGCBehaviorCloningNet(state_dim, num_actions).to(device)
    model.load_state_dict(torch.load("vgc_bc_model.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(X_test.to(device))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    # 4. Count frequencies for the top-10 ground truth moves
    true_counts = Counter(y_test)
    pred_counts = Counter(y_pred)
    total_test = len(y_test)

    top_10_ids = [move_id for move_id, _ in true_counts.most_common(10)]
    move_labels = [id_to_move[mid] for mid in top_10_ids]

    human_pct = [(true_counts[mid] / total_test) * 100.0 for mid in top_10_ids]
    model_pct = [(pred_counts[mid] / total_test) * 100.0 for mid in top_10_ids]

    # 5. Plot grouped bar chart
    x = np.arange(len(move_labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    ax.bar(
        x - width / 2,
        human_pct,
        width,
        label="Human Ground Truth",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        model_pct,
        width,
        label="BC Model Predictions",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title("Top-10 Move Distribution: Human Replays vs. BC Model (Test Set)")
    ax.set_ylabel("Share of Test Set Turns (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(move_labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/top10_move_frequency.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("figures/top10_move_frequency.png", format="png", bbox_inches="tight")
    print("Saved move frequency chart to figures/top10_move_frequency.pdf and figures/top10_move_frequency.png")
    plt.show()


if __name__ == "__main__":
    generate_move_frequency_plot()