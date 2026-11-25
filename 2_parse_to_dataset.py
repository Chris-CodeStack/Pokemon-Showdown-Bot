import os
import glob
import json
from collections import Counter
import numpy as np

RAW_DIR = "data/raw_replays"
OUTPUT_FILE = "data/processed_dataset.npz"
VOCAB_FILE = "data/vocab.json"
TOP_K_MOVES = 50
TOP_K_MONS = 30

def clean_mon_name(raw_str):
    """Extracts base Pokémon name, stripping level, gender, and nicknames."""
    name = raw_str.split(",")[0].strip()
    if ":" in name:
        name = name.split(":")[-1].strip()
    return name

def normalize_key(name):
    """Normalizes move/Pokémon names for reliable matching with poke-env."""
    return "".join(ch.lower() for ch in name if ch.isalnum())

def parse_hp(hp_str):
    """Converts Showdown HP string (e.g. '75/100', '0 fnt') to a float fraction."""
    if not hp_str or "fnt" in hp_str:
        return 0.0
    try:
        current, total = hp_str.split()[0].split("/")
        return float(current) / float(total)
    except Exception:
        return 1.0

def encode_species_one_hot(mon_name, mon_to_id):
    """Returns a (TOP_K_MONS + 1) one-hot vector; last index is 'Other/None'."""
    vec = [0.0] * (len(mon_to_id) + 1)
    idx = mon_to_id.get(normalize_key(mon_name), len(mon_to_id))
    vec[idx] = 1.0
    return vec

def process_replays():
    replay_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    if not replay_files:
        print(f"No replay files found in '{RAW_DIR}'. Run '1_download_replays.py' first!")
        return

    print(f"Found {len(replay_files)} replays. Parsing battle states and actions...")

    raw_samples = []
    mon_counter = Counter()
    move_counter = Counter()
    match_lengths = []

    for filepath in replay_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        log = data.get("log", "")
        lines = log.split("\n")

        # Active slots: [p1a, p1b, p2a, p2b]
        active_mons = ["None", "None", "None", "None"]
        active_hps = [1.0, 1.0, 1.0, 1.0]
        fainted_counts = {"p1": 0, "p2": 0}
        turn_count = 0

        for line in lines:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue

            cmd = parts[1]

            if cmd == "turn":
                turn_count += 1

            elif cmd in ("switch", "drag"):
                if len(parts) >= 5:
                    slot_tag = parts[2][:3]
                    mon_name = clean_mon_name(parts[3])
                    hp_val = parse_hp(parts[4])
                    mon_counter[normalize_key(mon_name)] += 1

                    idx_map = {"p1a": 0, "p1b": 1, "p2a": 2, "p2b": 3}
                    if slot_tag in idx_map:
                        idx = idx_map[slot_tag]
                        active_mons[idx] = mon_name
                        active_hps[idx] = hp_val

            elif cmd == "-damage":
                if len(parts) >= 4:
                    slot_tag = parts[2][:3]
                    hp_val = parse_hp(parts[3])
                    idx_map = {"p1a": 0, "p1b": 1, "p2a": 2, "p2b": 3}
                    if slot_tag in idx_map:
                        active_hps[idx_map[slot_tag]] = hp_val

            elif cmd == "-faint":
                if len(parts) >= 3:
                    slot_tag = parts[2][:3]
                    idx_map = {"p1a": 0, "p1b": 1, "p2a": 2, "p2b": 3}
                    if slot_tag in idx_map:
                        active_hps[idx_map[slot_tag]] = 0.0
                        side = slot_tag[:2]
                        fainted_counts[side] = min(4, fainted_counts[side] + 1)

            elif cmd == "move":
                if len(parts) >= 4:
                    actor = parts[2][:3]
                    move_key = normalize_key(parts[3].strip())
                    if not move_key:
                        continue
                    move_counter[move_key] += 1

                    # Normalize perspective so the model always sees [my_left, my_right, opp_left, opp_right]
                    if actor.startswith("p1"):
                        my_hps = active_hps[0:2]
                        opp_hps = active_hps[2:4]
                        my_mons = active_mons[0:2]
                        opp_mons = active_mons[2:4]
                        my_alive = (4 - fainted_counts["p1"]) / 4.0
                        opp_alive = (4 - fainted_counts["p2"]) / 4.0
                    else:
                        my_hps = active_hps[2:4]
                        opp_hps = active_hps[0:2]
                        my_mons = active_mons[2:4]
                        opp_mons = active_mons[0:2]
                        my_alive = (4 - fainted_counts["p2"]) / 4.0
                        opp_alive = (4 - fainted_counts["p1"]) / 4.0

                    actor_is_left = 1.0 if actor.endswith("a") else 0.0

                    raw_samples.append({
                        "hps": my_hps + opp_hps,
                        "turn": min(turn_count / 15.0, 1.0),
                        "alive": [my_alive, opp_alive],
                        "actor_is_left": actor_is_left,
                        "mons": my_mons + opp_mons,
                        "move": move_key,
                    })

        if turn_count > 0:
            match_lengths.append(turn_count)

    # Build vocabularies
    top_moves = [m for m, _ in move_counter.most_common(TOP_K_MOVES)]
    move_to_id = {m: idx for idx, m in enumerate(top_moves)}

    top_mons = [m for m, _ in mon_counter.most_common(TOP_K_MONS)]
    mon_to_id = {m: idx for idx, m in enumerate(top_mons)}

    filtered_X = []
    filtered_y = []

    for s in raw_samples:
        if s["move"] not in move_to_id:
            continue

        # Base numeric features: 4 HPs + 1 turn + 2 alive fractions + 1 actor slot = 8 features
        feat = list(s["hps"]) + [s["turn"]] + list(s["alive"]) + [s["actor_is_left"]]

        # Species one-hot features: 4 active slots * (TOP_K_MONS + 1)
        for mon in s["mons"]:
            feat.extend(encode_species_one_hot(mon, mon_to_id))

        filtered_X.append(feat)
        filtered_y.append(move_to_id[s["move"]])

    X = np.array(filtered_X, dtype=np.float32)
    y = np.array(filtered_y, dtype=np.int64)

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, X=X, y=y)

    vocab = {
        "move_to_id": move_to_id,
        "id_to_move": {str(idx): m for m, idx in move_to_id.items()},
        "mon_to_id": mon_to_id,
        "state_dim": int(X.shape[1]),
        "num_actions": len(top_moves),
    }
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print("\n" + "=" * 45)
    print("PARSING COMPLETE: ENRICHED DATASET SUMMARY")
    print("=" * 45)
    print(f"Replays Processed:       {len(match_lengths)}")
    print(f"Total Decision Samples:  {len(y)}")
    print(f"Feature Dimension (X):   {X.shape}")
    print(f"Action Vocabulary Size:  {len(top_moves)} moves")
    print(f"Species Vocabulary Size: {len(top_mons)} Pokémon (+1 Other)")
    print(f"Saved dataset to:        {OUTPUT_FILE}")
    print(f"Saved vocabularies to:   {VOCAB_FILE}")

if __name__ == "__main__":
    process_replays()