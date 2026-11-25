import json
import torch
import torch.nn as nn
from poke_env.player import DoubleBattleOrder, Player

def normalize_key(name):
    return "".join(ch.lower() for ch in name if ch.isalnum())

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
        return self.action_head(features)


class VGCBehaviorCloningBot(Player):
    def __init__(self, model_path="vgc_bc_model.pt", vocab_path="data/vocab.json", temperature=1.5, **kwargs):
        super().__init__(**kwargs)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.move_to_id = vocab["move_to_id"]
        self.mon_to_id = vocab["mon_to_id"]
        self.state_dim = vocab["state_dim"]
        self.num_actions = vocab["num_actions"]
        self.temperature = temperature

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VGCBehaviorCloningNet(self.state_dim, self.num_actions).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _encode_species(self, mon):
        vec = [0.0] * (len(self.mon_to_id) + 1)
        if mon is None:
            vec[-1] = 1.0
            return vec
        idx = self.mon_to_id.get(normalize_key(mon.species), len(self.mon_to_id))
        vec[idx] = 1.0
        return vec

    def _encode_state(self, battle, slot_idx):
        # Active slots: [my_left, my_right, opp_left, opp_right]
        my_active = list(battle.active_pokemon)
        opp_active = list(battle.opponent_active_pokemon)

        # Pad to 2 slots if needed
        while len(my_active) < 2:
            my_active.append(None)
        while len(opp_active) < 2:
            opp_active.append(None)

        my_hps = [m.current_hp_fraction if (m and not m.fainted) else 0.0 for m in my_active[:2]]
        opp_hps = [m.current_hp_fraction if (m and not m.fainted) else 0.0 for m in opp_active[:2]]

        turn_feat = min(battle.turn / 15.0, 1.0)

        my_alive = sum(1 for m in battle.team.values() if not m.fainted) / 4.0
        opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
        opp_alive = max(0, 4 - opp_fainted) / 4.0

        actor_is_left = 1.0 if slot_idx == 0 else 0.0

        feat = my_hps + opp_hps + [turn_feat, min(my_alive, 1.0), opp_alive, actor_is_left]
        for mon in my_active[:2] + opp_active[:2]:
            feat.extend(self._encode_species(mon))

        return torch.tensor([feat], dtype=torch.float32, device=self.device)

    def _pick_masked_move(self, battle, slot_idx):
        available_moves = battle.available_moves[slot_idx]
        if not available_moves:
            return None

        state_tensor = self._encode_state(battle, slot_idx)
        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0) / self.temperature

        # Mask out all moves not currently legal for this slot
        masked_logits = torch.full_like(logits, float("-inf"))
        id_to_legal_move = {}

        for move in available_moves:
            m_key = normalize_key(move.id)
            if m_key in self.move_to_id:
                action_id = self.move_to_id[m_key]
                masked_logits[action_id] = logits[action_id]
                id_to_legal_move[action_id] = move

        if not id_to_legal_move:
            # Fallback if none of the legal moves are in the top-50 vocabulary
            return available_moves[0]

        best_action_id = int(torch.argmax(masked_logits).item())
        return id_to_legal_move[best_action_id]

    def choose_move(self, battle):
        orders = [None, None]
        used_switches = set()

        for slot_idx in range(2):
            if battle.force_switch[slot_idx]:
                switches = [
                    mon for mon in battle.available_switches[slot_idx]
                    if mon not in used_switches
                ]
                if switches:
                    used_switches.add(switches[0])
                    orders[slot_idx] = self.create_order(switches[0])
                continue

            chosen_move = self._pick_masked_move(battle, slot_idx)
            mon = battle.active_pokemon[slot_idx]
            if chosen_move is not None and mon is not None:
                targets = battle.get_possible_showdown_targets(chosen_move, mon)
                # Prefer targeting an opposing Pokémon (> 0) if available
                foe_targets = [t for t in targets if t > 0]
                target = foe_targets[0] if foe_targets else (targets[0] if targets else 0)
                orders[slot_idx] = self.create_order(chosen_move, move_target=target)

        if orders[0] is None or orders[1] is None:
            return self.choose_random_doubles_move(battle)
        return DoubleBattleOrder(orders[0], orders[1])