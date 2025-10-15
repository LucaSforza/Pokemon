import json
import sys
from typing import Any

PokemonName = str
Level = int

def main():
    paths: list[str] = sys.argv[1:]
    pokemons: dict[PokemonName, dict[Level, Any]] = {}
    for path in paths:
        with open(path) as f:
            for line in f.readlines():
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
    with open("pokemons.json","w") as f:
        f.write(json.dumps(pokemons, indent=2))
    for key in pokemons.keys():
        print(key)
    print(len(pokemons.keys()))

def main2():
    paths: list[str] = sys.argv[1:]
    pokemons: dict[PokemonName, dict[Level, Any]] = {}
    for path in paths:
        with open(path) as f:
            for line in f.readlines():
                battle: dict[str, Any] = json.loads(line)
                team: list[dict[str, Any]] = battle["p1_team_details"]
                for pokemon in team:
                    pokemonName = pokemon["name"]
                    level = pokemon["level"]
                    if pokemons.get(pokemonName) is None:
                        pokemons[pokemonName] = {}
                    if pokemons[pokemonName].get(level) is None:
                        pokemons[pokemonName][level] = {}
                        pokemon.__delitem__("name")
                        pokemon.__delitem__("level")
                        pokemons[pokemonName][level].update(pokemon)
    with open("pokemons.json","w") as f:
        f.write(json.dumps(pokemons, indent=2))
    for key in pokemons.keys():
        print(key)
    print(len(pokemons.keys()))

if __name__ == "__main__":
    main()