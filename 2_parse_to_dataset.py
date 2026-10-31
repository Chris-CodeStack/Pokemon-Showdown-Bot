import json
import os
import numpy as np

# In VGC Doubles, there are 4 active slots on the field:
# Slot 0: P1 Left (p1a), Slot 1: P1 Right (p1b)
# Slot 2: P2 Left (p2a), Slot 3: P2 Right (p2b)

def parse_replay_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    log = data.get("log", "")
    lines = log.split("\n")
    
    samples = []
    # Simplified turn parser:
    current_state = {
        "p1_active": ["", ""],
        "p2_active": ["", ""],
        "p1_hp": [1.0, 1.0],
        "p2_hp": [1.0, 1.0]
    }
    
    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
            
        cmd = parts[1]
        
        # Track active switch-ins
        if cmd == "switch":
            slot = parts[2][:3]
            mon = parts[2][5:]
            if slot == "p1a": current_state["p1_active"][0] = mon
            elif slot == "p1b": current_state["p1_active"][1] = mon
            elif slot == "p2a": current_state["p2_active"][0] = mon
            elif slot == "p2b": current_state["p2_active"][1] = mon
            
        # Extract actions chosen by players
        elif cmd == "move":
            actor = parts[2][:3]
            move_name = parts[3]
            target = parts[4] if len(parts) > 4 else "none"
            
    return samples