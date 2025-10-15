import json
import sys
from typing import Any


POKEMONS_PATH = "pokemons.json"

def main():
    file    = sys.argv[1]
    team_id = int(sys.argv[2])
    battles = []
    print(f"Loading {file}...")
    with open(file, "r") as f:
        for line in f:
            battles.append(json.loads(line))
    pokemons: dict[str, Any] = None
    with open(POKEMONS_PATH, "r") as f:
        pokemons = json.load(f)
    
    battle = battles[team_id]
    team1 = []
    for p in battle["p1_team_details"]:
        pokemon = pokemons[p["name"]]
        pokemon["name"] = p["name"]
        team1.append(pokemon)
    team2 = []
    pokemon = pokemons[battle["p2_lead_details"]["name"]]
    pokemon["name"] = battle["p2_lead_details"]["name"]
    team2.append(pokemon)
    for t in battle["battle_timeline"]:
        state_pokemon = t["p2_pokemon_state"]
        pokemon = pokemons[state_pokemon["name"]]
        pokemon["name"] = state_pokemon["name"]
        if pokemon not in team2:
            team2.append(pokemon)
    print(f"Team 1: len: {len(team1)}\n{json.dumps(team1, indent=2)}")
    print(f"Team 2: len: {len(team2)}\n{json.dumps(team2, indent=2)}")


if __name__ == "__main__":
    main()