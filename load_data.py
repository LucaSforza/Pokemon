import sys
import sqlite3
import json

from data_analyzer import *


def log_open_file(path):
    print(f"Loading {path}...")


def load_pokemon(cur: sqlite3.Cursor, pokemon: Pokemon):
    fields = ("name", "base_hp", "base_atk", "base_def",
              "base_spa", "base_spd", "base_spe")
    values = tuple(pokemon[f] for f in fields)
    cur.execute(
        f"INSERT OR IGNORE INTO Pokemon ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)


def insert_state_move(cur: sqlite3.Cursor, state: State, move: Move) -> int:

    move_name = None
    try:
        move_name = move["name"]
    except TypeError:
        pass

    if move_name is not None:
        fields = ("name", "pokemon", "base_power",
                  "accuracy", "priority", "type", "category")
        values = (move_name, state["name"], move["base_power"],
                  move["accuracy"], move["priority"], move["type"], move["category"])

        cur.execute(
            f"INSERT OR IGNORE INTO MoveType (name) VALUES (?)", (move["type"],))
        cur.execute(
            f"INSERT OR IGNORE INTO MoveCategory (name) VALUES (?)", (move["category"],))

        cur.execute(
            f"INSERT OR IGNORE INTO PokemonMove ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)

    cur.execute(f"INSERT OR IGNORE INTO Status (name) VALUES (?)",
                (state["status"],))

    fields = ("hp_pct", "status", "boost_atk", "boost_def", "boost_spa",
              "boost_spd", "boost_spe", "pokemon", "pok_move")
    values = (state["hp_pct"], state["status"], state["boosts"]["atk"], state["boosts"]["def"], state["boosts"]
              ["spa"], state["boosts"]["spd"], state["boosts"]["spe"], state["name"], move_name)
    cur.execute(
        f"INSERT INTO PokemonState ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)
    state_id = cur.lastrowid
    for effect in state["effects"]:
        cur.execute(
            f"INSERT OR IGNORE INTO Effect (name) VALUES (?)", (effect,))
    
        cur.execute(
            f"INSERT INTO eff_pok (effect, pok_state) VALUES (?,?)", (effect,state_id))


    return state_id


def insert_turn(cur: sqlite3.Cursor, turn: Turn, battle_id : int):
    turn_id = turn["turn"]
    p1_move_details: Move | None = turn["p1_move_details"]
    p1_pokemon_state: State = turn["p1_pokemon_state"]
    p2_move_details: Move | None = turn["p2_move_details"]
    p2_pokem_state: State = turn["p2_pokemon_state"]

    p1_state_id = insert_state_move(cur, p1_pokemon_state, p1_move_details)
    p2_state_id = insert_state_move(cur, p2_pokem_state, p2_move_details)

    fields = ("id", "battle", "p1_state", "p2_state")
    values = (turn_id, battle_id, p1_state_id, p2_state_id)

    cur.execute(
        f"INSERT INTO Turn ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)



def load_team(cur: sqlite3.Cursor, pokemon: Pokemon, id_battle: int):
    fields = ("pokemon", "battle", "level")
    values = (pokemon["name"], id_battle, pokemon["level"])
    cur.execute(
        f"INSERT INTO TeamP1 ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)


def insert_battle(cur: sqlite3.Cursor, battle: dict) -> int:
    player_won: bool | None = None
    try:
        player_won = battle["player_won"]
    except KeyError:
        pass
    pokemons: list[Pokemon] = battle["p1_team_details"]
    for pokemon in pokemons:
        load_pokemon(cur, pokemon)
    fields = ("battle_id", "result", "p2_lead_pokemon", "p2_pokeon_level")
    cur.execute(
        f"INSERT INTO Battle ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
        (battle["battle_id"], player_won, battle["p2_lead_details"]
         ["name"], battle["p2_lead_details"]["level"])
    )
    battle_id = cur.lastrowid
    for turn in battle["battle_timeline"]:
        insert_turn(cur, turn, battle_id)

    for pokemon in pokemons:
        load_team(cur, pokemon, battle_id)
    return battle_id
    

def load_dataset(db: sqlite3.Connection, path: str, training: bool):
    cur = db.cursor()

    log_open_file(path)
    type = "Train" if training else "Test"
    cur.execute("INSERT INTO Dataset (type) VALUES (?)", (type,))
    dataset_id = cur.lastrowid
    with open(path, "r") as f:
        print(path)
        for battle in f:
            battle: Battle = json.loads(battle)
            battle_id = insert_battle(cur, battle)
            if battle.get("player_won") is not None == training:
                print("[FATAL ERROR] the entry in not in the right dataset")
                exit(1)
            cur.execute("INSERT INTO bat_dat (dataset,battle) VALUES (?,?)", (dataset_id, battle_id))
    db.commit()


def main():
    # Creazione della connessione
    database_path = sys.argv[1]
    train_set = sys.argv[2]
    test_set = sys.argv[3]
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        load_dataset(conn, train_set, True)
        load_dataset(conn, test_set, False)


if __name__ == "__main__":
    main()
