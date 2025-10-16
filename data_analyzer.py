
import sys
from typing import Any
import json

PROGRAM_NAME = sys.argv[0]

Pokemon = dict[str, Any]
PokemonDict = dict[str, Any]
Team = list[Pokemon]
Battle = dict[str, Any]
Turn = dict[str, Any]
State = dict[str, Any]
Move = dict[str, Any]

class DataProcessor:
    pass

def print_team(team: Team) -> None:
    print(f"Team: len: {len(team)}\n{json.dumps(team, indent=2)}")

def get_pokemons(paths: list[str]) -> PokemonDict:
    pokemons: PokemonDict = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                battle: dict[str, Any] = json.loads(line)
                team: list[dict[str, Any]] = battle["p1_team_details"]
                for pokemon in team:
                    pokemonName = pokemon["name"]
                    _level = pokemon["level"]
                    if pokemons.get(pokemonName) is None:
                        pokemons[pokemonName] = {}
                        pokemon.__delitem__("name")
                        pokemon.__delitem__("level")
                        pokemons[pokemonName].update(pokemon)
    return pokemons

def get_teams(battle: Battle, pokemons: PokemonDict) -> tuple[Team, Team]:
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
    return team1, team2

def load_pokemons(path: str) -> PokemonDict:
    with open(path, "r") as f:
        pokemons = json.load(f)
    return pokemons

def usage():
    print(f"{PROGRAM_NAME} <command> [args...]")
    print("    get_pokemons <outputfile> <datasets...> salva nel file di output tutti i pokemon che trovi")
    print("    get_teams <outputfile> <dataset> <game id> <pokemon_file> prendi le squadre che trovi di un certo id")

def main():
    command = sys.argv[1]
    if command == "get_pokemons":
        try:
            output_file = sys.argv[2]
            paths = sys.argv[3:]
        except IndexError:
            print("[ERROR] not enough arguments")
            usage()
            exit(1)
        pokemons = get_pokemons(paths)
        with open(output_file, "w") as f:
            f.write(json.dumps(pokemons, indent=2))
    elif command == "get_teams":
        try:
            file = sys.argv[2]
            game_id = int(sys.argv[3])
            pokemons_path = sys.argv[4]
        except IndexError:
            print("[ERROR] not enough arguments")
            usage()
            exit(1)
        except ValueError:
            print("[ERROR] the game id is not an integer")
            usage()
            exit(1)
        battles = []
        print(f"Loading {file}...")
        with open(file, "r") as f:
            for line in f:
                battles.append(json.loads(line))
        battle = battles[game_id]
        pokemons: PokemonDict = load_pokemons(pokemons_path)
        (team1, team2) = get_teams(battle, pokemons)
        print_team(team1)
        print_team(team2)
    else:
        print(f"[ERROR] unknown command {command}")
        usage()
        exit(1)

if __name__ == "__main__":
    main()