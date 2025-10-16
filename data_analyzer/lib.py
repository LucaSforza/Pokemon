import json
import sqlite3
from typing import Any


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

def get_pokemons(cur: sqlite3.Cursor) -> list[Pokemon]:
    cur.execute("SELECT * FROM Pokemon")
    return into_dict(cur)

def get_teams(cur: sqlite3.Cursor, game_id: int, training: bool) -> tuple[Team, Team]:
    pass