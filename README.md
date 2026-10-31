# Pokémon Showdown AI Battle Bot

A machine learning and heuristic-driven battle bot designed to make tactical, turn-by-turn decisions in competitive Pokémon Showdown battles using visible state observations.

### 1. Set Up the Local Pokémon Showdown Server

A local server prevents network latency, bypasses online challenge rate-limits, and accelerates evaluation runs.

```bash
# Clone the Pokémon Showdown engine
git clone [https://github.com/smogon/pokemon-showdown.git](https://github.com/smogon/pokemon-showdown.git)

# Enter directory and install dependencies
cd pokemon-showdown
npm install

# Start the server locally without security restrictions
node pokemon-showdown start --no-security
```
The server will now be listening locally at `http://localhost:8000`. Keep this terminal window open.

---

### 2. Configure Python & `poke-env`

In a separate terminal, set up the Python environment:

```bash
# Optional: create and activate a virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install poke-env
pip install poke-env
```

---

## Running the Baseline Bot

Once the local server is running, execute the baseline script:

```bash
python bot.py
```

### Expected Behavior
- Connects to the local server at `localhost:8000`.
- Automatically initializes two players (`SmartHeuristicBot` and `RandomPlayer`).
- Runs batch battles in the `gen9randombattle` format.
- Prints the total finished battles and the resulting win-rate percentage.

---

## Repository Structure

- `bot.py` — Heuristic decision bot implementation and battle execution script.
- `README.md` — Setup, installation, and environment documentation.
- `.gitignore` — Ignore rules for Node packages, Python artifacts, and runtime caches.