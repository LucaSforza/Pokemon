import json
import sqlite3
from typing import Any
import numpy as np
import pandas as pd

Pokemon = dict[str, Any]
PokemonDict = dict[str, Any]
Team = list[Pokemon]
Battle = dict[str, Any]
Turn = dict[str, Any]
State = dict[str, Any]
Move = dict[str, Any]

def print_team(team: Team) -> None:
    print(f"Team: len: {len(team)}\n{json.dumps(team, indent=2)}")

def into_dict(cur: sqlite3.Cursor) -> list[dict]:
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in rows]

def into_dataframe(cur: sqlite3.Cursor) -> pd.DataFrame:
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)

def get_pokemons(cur: sqlite3.Cursor) -> list[Pokemon]:
    cur.execute("SELECT * FROM Pokemon")
    return into_dataframe(cur)

def get_teams(cur: sqlite3.Cursor, game_id: int, _set: str) -> pd.DataFrame:
    cur.execute("SELECT id FROM Dataset WHERE id = ?", (_set,))
    id_battle = cur.lastrowid
    cur.execute("SELECT * FROM Battle WHERE id = ?", (id_battle,))
    battle = into_dataframe(cur)
    
    battle_id: int = battle["battle_id"]
    team_id = battle["team"]
    cur.execute("""
    SELECT name,base_hp,base_atk, base_def, base_spa,base_spd, base_spe 
    FROM Level as l,Pokemon as p
    WHERE team = ? and l.pokemon = p.name
        """)
    team1 = into_dataframe(cur)
    
    cur.execute("""
    SELECT p.name,p.base_hp,p.base_atk, p.base_def, p.base_spa,p.base_spd, p.base_spe 
    FROM Turn as t, PokemonState as ps, Pokemon as p
    WHERE 
    """)
    
    
    pass