import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class VGCBehaviorCloningNet(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, num_actions)

    def forward(self, state):
        features = self.encoder(state)
        logits = self.action_head(features)
        return logits


def compute_accuracies(logits, targets, k=3):
    with torch.no_grad():
        # Top-1 Accuracy
        preds_top1 = torch.argmax(logits, dim=1)
        top1_acc = (preds_top1 == targets).float().mean().item() * 100.0

        # Top-k Accuracy
        actual_k = min(k, logits.shape[1])
        _, topk_preds = logits.topk(actual_k, dim=1, largest=True, sorted=True)
        topk_correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
        topk_acc = topk_correct.any(dim=1).float().mean().item() * 100.0

        return top1_acc, topk_acc


def run_training():
    dataset_path = "data/processed_dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run '2_parse_to_dataset.py' first.")
        return

    # Load dataset
    data = np.load(dataset_path)
    X = data["X"]
    y = data["y"]

    num_samples, state_dim = X.shape
    num_actions = int(np.max(y) + 1)

    print("=" * 55)
    print("VGC BEHAVIOR CLONING PIPELINE")
    print("=" * 55)
    print(f"Total Samples:     {num_samples}")
    print(f"State Dimension:   {state_dim}")
    print(f"Unique Actions:    {num_actions}")

    # Set device & seed
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device:   {device}")

    # 70% Train / 15% Validation / 15% Test Split
    indices = np.random.permutation(num_samples)
    train_end = int(0.70 * num_samples)
    val_end = int(0.85 * num_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print(f"Split Breakdown:   Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

    # Convert to PyTorch Tensors
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)

    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    # Initialize model, loss, optimizer
    model = VGCBehaviorCloningNet(state_dim=state_dim, num_actions=num_actions).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 20
    print("\nStarting Training...\n")
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Top-1 (%)':<15} | {'Val Top-3 (%)':<15}")
    print("-" * 65)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss_accum = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * len(batch_y)

        train_loss = train_loss_accum / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_logits_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_accum += loss.item() * len(batch_y)
                val_logits_list.append(logits)
                val_targets_list.append(batch_y)

        val_loss = val_loss_accum / len(val_loader.dataset)
        val_top1, val_top3 = compute_accuracies(
            torch.cat(val_logits_list), torch.cat(val_targets_list), k=3
        )

        print(f"{epoch:<6} | {train_loss:<12.4f} | {val_loss:<10.4f} | {val_top1:<15.2f} | {val_top3:<15.2f}")

        # Save checkpoint if best validation score
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "vgc_bc_model.pt")

    print("\n" + "=" * 55)
    print("FINAL TEST EVALUATION (vgc_bc_model.pt)")
    print("=" * 55)
    model.load_state_dict(torch.load("vgc_bc_model.pt"))
    model.eval()

    test_loss_accum = 0.0
    test_logits_list = []
    test_targets_list = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            test_loss_accum += criterion(logits, batch_y).item() * len(batch_y)
            test_logits_list.append(logits)
            test_targets_list.append(batch_y)

    test_loss = test_loss_accum / len(test_loader.dataset)
    test_top1, test_top3 = compute_accuracies(
        torch.cat(test_logits_list), torch.cat(test_targets_list), k=3
    )

    print(f"Test Loss:             {test_loss:.4f}")
    print(f"Test Top-1 Accuracy:   {test_top1:.2f}%")
    print(f"Test Top-3 Accuracy:   {test_top3:.2f}%")
    print("Checkpoint saved as 'vgc_bc_model.pt'.")


if __name__ == "__main__":
    run_training()