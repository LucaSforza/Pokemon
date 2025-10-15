import sys
import sqlite3
import json

from data_analyzer import *

def log_open_file(path):
    print(f"Loading {path}...")

def load_pokemon(cur: sqlite3.Cursor, pokemon: Pokemon):
    fields = ("name", "base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe")
    values = tuple(pokemon[f] for f in fields)
    cur.execute(f"INSERT OR IGNORE INTO Pokemon ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)

def insert_battle(cur: sqlite3.Cursor, battle: dict):
    player_won: bool | None = None
    try:
        player_won = battle["player_won"]
    except KeyError:
        pass
    pokemons: list[Pokemon] = battle["p1_team_details"]
    for pokemon in pokemons:
        load_pokemon(cur, pokemon)
    cur.execute("INSERT INTO Team DEFAULT VALUES")
    team_id = cur.lastrowid
    fields = ("result","player","p2_lead_pokemon","p2_pokeon_level")
    cur.execute(
        f"INSERT INTO Battle ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
        (player_won, team_id, battle["p2_lead_pokemon"]["name"], battle["p2_lead_pokemon"]["level"])
    )
    

def load_train_dataset(db: sqlite3.Connection, path: str):
    cur = db.cursor()
    
    log_open_file(path)
    
    with open(path, "r") as f:
        print(path)
        for battle in f:
            insert_battle(cur, json.loads(battle))
    db.commit()

def load_test_dataset(db: sqlite3.Connection, path: str):
    pass

def main():
    # Creazione della connessione
    database_path = sys.argv[1]
    train_set = sys.argv[2]
    test_set = sys.argv[3]
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        load_train_dataset(conn,train_set)
        load_test_dataset(conn,test_set)    
    
if __name__ == "__main__":
    main()