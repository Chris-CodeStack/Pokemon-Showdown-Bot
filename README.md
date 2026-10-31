# Pokémon Showdown AI Battle Bot

A machine learning and heuristic-driven battle bot designed to make tactical, turn-by-turn decisions in competitive Pokémon Showdown battles using visible state observations.

### 1. Set Up the Local Pokémon Showdown Server

A local server prevents network latency, bypasses online challenge rate-limits, and accelerates evaluation runs.

```bash
pip install poke-env
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```
The server will now be listening locally at `http://localhost:8000`. Keep this terminal window open.

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
