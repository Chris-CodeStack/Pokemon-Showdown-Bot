import asyncio
from poke_env import LocalhostServerConfiguration
from poke_env.player import Player, RandomPlayer

class SmartHeuristicBot(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            # Score = base_power * type_multiplier (accounting for STAB and immunities)
            def move_score(move):
                if move.category.name == "STATUS":
                    return 0
                multiplier = battle.opponent_active_pokemon.damage_multiplier(move)
                stab = 1.5 if move.type in battle.active_pokemon.types else 1.0
                return move.base_power * multiplier * stab

            best_move = max(battle.available_moves, key=move_score)
            return self.create_order(best_move)
        
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_random_move(battle)
        

async def main():
    bot = SmartHeuristicBot(
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )
    opponent = RandomPlayer(
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )

    print("Starting evaluation battles against RandomPlayer...")
    
    await bot.battle_against(opponent, n_battles=100)

    print(f"Battles finished: {bot.n_finished_battles}")
    print(f"Bot win rate: {bot.n_won_battles / bot.n_finished_battles * 100:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())