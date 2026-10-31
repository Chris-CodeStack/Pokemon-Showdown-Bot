import os
import glob
import json
from collections import Counter
import numpy as np

RAW_DIR = "data/raw_replays"
OUTPUT_FILE = "data/processed_dataset.npz"

def clean_mon_name(raw_str):
    name = raw_str.split(",")[0].strip()
    if ":" in name:
        name = name.split(":")[-1].strip()
    return name

def parse_hp(hp_str):
    if not hp_str or "fnt" in hp_str:
        return 0.0
    try:
        current, total = hp_str.split()[0].split("/")
        return float(current) / float(total)
    except Exception:
        return 1.0

def process_replays():
    replay_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    if not replay_files:
        print(f"No replay files found in '{RAW_DIR}'. Run '1_download_replays.py' first!")
        return

    print(f"Found {len(replay_files)} replays. Parsing battle states and actions...")

    all_states = []
    all_actions = []
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

        # Track active slots: [P1_Left, P1_Right, P2_Left, P2_Right]
        active_mons = ["None", "None", "None", "None"]
        active_hps = [1.0, 1.0, 1.0, 1.0]
        
        turn_count = 0
        current_turn_moves = []

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
                    mon_counter[mon_name] += 1

                    if slot_tag == "p1a":
                        active_mons[0], active_hps[0] = mon_name, hp_val
                    elif slot_tag == "p1b":
                        active_mons[1], active_hps[1] = mon_name, hp_val
                    elif slot_tag == "p2a":
                        active_mons[2], active_hps[2] = mon_name, hp_val
                    elif slot_tag == "p2b":
                        active_mons[3], active_hps[3] = mon_name, hp_val

            elif cmd == "-damage":
                # Format: |-damage|p1a: Incineroar|45/100
                if len(parts) >= 4:
                    slot_tag = parts[2][:3]
                    hp_val = parse_hp(parts[3])
                    idx_map = {"p1a": 0, "p1b": 1, "p2a": 2, "p2b": 3}
                    if slot_tag in idx_map:
                        active_hps[idx_map[slot_tag]] = hp_val

            elif cmd == "-faint":
                # Format: |-faint|p1a: Incineroar
                if len(parts) >= 3:
                    slot_tag = parts[2][:3]
                    idx_map = {"p1a": 0, "p1b": 1, "p2a": 2, "p2b": 3}
                    if slot_tag in idx_map:
                        active_hps[idx_map[slot_tag]] = 0.0

            elif cmd == "move":
                # Format: |move|p1a: Urshifu|Wicked Blow|p2a: Calyrex
                if len(parts) >= 4:
                    actor = parts[2][:3]
                    move_name = parts[3].strip()
                    target = parts[4].strip() if len(parts) > 4 else "none"
                    move_counter[move_name] += 1

                    # Record decision sample
                    # Feature representation: current 4 active HP values + turn count
                    state_vec = list(active_hps) + [min(turn_count / 15.0, 1.0)]
                    all_states.append(state_vec)
                    all_actions.append(move_name)

        if turn_count > 0:
            match_lengths.append(turn_count)

    # Build action vocabulary mapping the top 50 moves to classification indices
    top_moves = [move for move, _ in move_counter.most_common(50)]
    move_to_id = {move: idx for idx, move in enumerate(top_moves)}

    # Filter dataset to valid mapped actions
    filtered_X = []
    filtered_y = []
    for s, a in zip(all_states, all_actions):
        if a in move_to_id:
            filtered_X.append(s)
            filtered_y.append(move_to_id[a])

    X = np.array(filtered_X, dtype=np.float32)
    y = np.array(filtered_y, dtype=np.int64)

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, X=X, y=y)

    print("\n" + "=" * 45)
    print("PARSING COMPLETE: DATASET SUMMARY")
    print("=" * 45)
    print(f"Replays Processed:       {len(match_lengths)}")
    print(f"Total Decision Samples:  {len(y)}")
    print(f"Feature Dimension (X):   {X.shape}")
    print(f"Action Vocabulary Size:  {len(top_moves)} moves")
    if match_lengths:
        print(f"Average Battle Turns:    {np.mean(match_lengths):.2f} turns (max: {max(match_lengths)})")
    
    print("\n--- Top 5 Most Common Pokémon ---")
    for mon, count in mon_counter.most_common(5):
        print(f"  {mon:<20}: {count} appearances")

    print("\n--- Top 5 Most Common Moves ---")
    for move, count in move_counter.most_common(5):
        print(f"  {move:<20}: {count} uses")
    
    print(f"\nSaved processed dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_replays()