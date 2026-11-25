import asyncio
import time
import numpy as np
from poke_env import AccountConfiguration
from poke_env.player import DoubleBattleOrder, Player, RandomPlayer
from bot import VGCBehaviorCloningBot

BATTLE_FORMAT = "gen9randomdoublesbattle"
NUM_BATTLES = 20


class GreedyDoublesBot(Player):
    """Simple heuristic baseline: picks the highest base-power legal move per slot."""

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

            available_moves = battle.available_moves[slot_idx]
            mon = battle.active_pokemon[slot_idx]
            if available_moves and mon is not None:
                best_move = max(available_moves, key=lambda m: m.base_power)
                targets = battle.get_possible_showdown_targets(best_move, mon)
                foe_targets = [t for t in targets if t > 0]
                target = foe_targets[0] if foe_targets else (targets[0] if targets else 0)
                orders[slot_idx] = self.create_order(best_move, move_target=target)

        if orders[0] is None or orders[1] is None:
            return self.choose_random_doubles_move(battle)
        return DoubleBattleOrder(orders[0], orders[1])


async def run_matchup(bot, opponent, label, n_battles):
    bot.reset_battles()
    opponent.reset_battles()

    print(f"Running {n_battles} battles: BC Bot vs. {label}...")
    await bot.battle_against(opponent, n_battles=n_battles)

    wins = bot.n_won_battles
    win_rate = (wins / n_battles) * 100.0
    avg_turns = np.mean([b.turn for b in bot.battles.values()]) if bot.battles else 0.0

    return {
        "opponent": label,
        "wins": wins,
        "total": n_battles,
        "win_rate": win_rate,
        "avg_turns": avg_turns,
    }


async def main():
    tag = str(int(time.time()))[-4:]

    bc_bot = VGCBehaviorCloningBot(
        model_path="vgc_bc_model.pt",
        vocab_path="data/vocab.json",
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration(f"BCBot_{tag}", None),
    )
    random_bot = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration(f"RandBot_{tag}", None),
    )
    greedy_bot = GreedyDoublesBot(
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration(f"GreedyBot_{tag}", None),
    )

    results = []
    results.append(await run_matchup(bc_bot, random_bot, "Random Baseline", NUM_BATTLES))
    results.append(await run_matchup(bc_bot, greedy_bot, "Greedy Damage Baseline", NUM_BATTLES))

    print("\n" + "=" * 62)
    print(f"LIVE BENCHMARK RESULTS ({BATTLE_FORMAT}, {NUM_BATTLES} matches each)")
    print("=" * 62)
    print(f"{'Opponent':<25} | {'Wins':<8} | {'Win Rate':<10} | {'Avg Turns':<10}")
    print("-" * 62)
    for r in results:
        print(
            f"{r['opponent']:<25} | {r['wins']}/{r['total']:<5} | {r['win_rate']:>7.1f}%  | {r['avg_turns']:>8.2f}"
        )
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())