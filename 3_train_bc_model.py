import torch
import torch.nn as nn

class VGCBehaviorCloningNet(nn.Module):
    def __init__(self, state_dim, num_actions=17):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Action head for Pokémon 1 (Left slot)
        self.head_slot_a = nn.Linear(128, num_actions)
        # Action head for Pokémon 2 (Right slot)
        self.head_slot_b = nn.Linear(128, num_actions)

    def forward(self, state):
        features = self.encoder(state)
        logits_a = self.head_slot_a(features)
        logits_b = self.head_slot_b(features)
        return logits_a, logits_b