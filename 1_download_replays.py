import os
import json
import time
import requests

# Set VGC format; fallback to gen9vgc2024regg or gen9doublesou if needed
FORMAT = "gen9vgc2024regg"
REPLAY_SEARCH_URL = f"https://replay.pokemonshowdown.com/search.json?format={FORMAT}&page="

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CS412ShowdownResearchBot/1.0"
}

os.makedirs("data/raw_replays", exist_ok=True)

replays = []
print(f"Searching for VGC replays ({FORMAT})...")

for page in range(1, 4): 
    try:
        resp = requests.get(f"{REPLAY_SEARCH_URL}{page}", headers=HEADERS)
        if resp.status_code != 200:
            print(f"Page {page} returned status {resp.status_code}, skipping.")
            continue
            
        items = resp.json()
        if not items:
            print(f"No replays found on page {page}.")
            break

        for item in items:
            rating = item.get("rating")
            rating_val = rating if isinstance(rating, (int, float)) else 0
            
            if rating_val >= 1300 or rating is None:
                replays.append(item["id"])

        print(f"Page {page}: found {len(items)} items. Total collected so far: {len(replays)}")
        time.sleep(1)
    except Exception as e:
        print(f"Error fetching page {page}: {e}")

print(f"\nCollected {len(replays)} replay IDs. Downloading full logs...")

downloaded_count = 0
for i, replay_id in enumerate(replays):
    filepath = f"data/raw_replays/{replay_id}.json"
    
    # Skip if already downloaded
    if os.path.exists(filepath):
        downloaded_count += 1
        continue

    try:
        log_url = f"https://replay.pokemonshowdown.com/{replay_id}.json"
        log_resp = requests.get(log_url, headers=HEADERS)
        
        if log_resp.status_code == 200:
            data = log_resp.json()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f)
            downloaded_count += 1
            if downloaded_count % 10 == 0:
                print(f"Downloaded {downloaded_count}/{len(replays)} replays...")
        time.sleep(0.5)  # Rate limiting courtesy pause
    except Exception as e:
        print(f"Failed to fetch {replay_id}: {e}")

print(f"\nDone! Successfully saved {downloaded_count} replays in 'data/raw_replays/'.")