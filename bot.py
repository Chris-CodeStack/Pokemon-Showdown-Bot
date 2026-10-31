import asyncio
from poke_env import LocalhostServerConfiguration
from poke_env.player import Player, RandomPlayer

class SmartHeuristicBot(Player):
    def choose_move(self, battle):
        # 1. Check if we can attack
        if battle.available_moves:
            # Simple heuristic: Pick the move with the highest base power
            best_move = max(battle.available_moves, key=lambda m: m.base_power)
            return self.create_order(best_move)
        
        # 2. If no moves are available (e.g., forced switch), switch to the first bench option
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
            
        return self.choose_random_move(battle)

async def main():
    # Configure players to connect to the local server running on localhost:8000
    bot = SmartHeuristicBot(
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )
    opponent = RandomPlayer(
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )

    print("Starting evaluation battles against RandomPlayer...")
    
    # Run 5 battles against the baseline opponent
    await bot.battle_against(opponent, n_battles=5)

    print(f"Battles finished: {bot.n_finished_battles}")
    print(f"Bot win rate: {bot.n_won_battles / bot.n_finished_battles * 100:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())