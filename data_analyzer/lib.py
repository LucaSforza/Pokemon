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

def get_teams(cur: sqlite3.Cursor, game_id: int, _set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cur.execute("SELECT id FROM Dataset WHERE type = ?", (_set,))
    id_dataset = cur.fetchone()[0]
    cur.execute("SELECT b.id FROM bat_dat, Battle as b WHERE dataset = ? and b.id = bat_dat.battle and b.battle_id = ?", (id_dataset,game_id))
    id_battle = cur.fetchone()[0]
    
    cur.execute("""
    SELECT p.name,p.base_hp,p.base_atk, p.base_def, p.base_spa,p.base_spd, p.base_spe
    FROM Level as l,Pokemon as p, Battle as b, Team as t
    WHERE t.id = b.team and l.team = t.id and l.pokemon = p.name and b.id = ?
        """, (id_battle,))
    team1 = into_dataframe(cur)
    
    cur.execute("""
    SELECT p.name,p.base_hp,p.base_atk, p.base_def, p.base_spa,p.base_spd, p.base_spe 
    FROM Turn as t, PokemonState as ps, Pokemon as p, Battle as b
    WHERE t.battle = ? and b.id = t.battle and t.p2_state = ps.id and (ps.pokemon = p.name or b.p2_lead_pokemon = p.name)
    GROUP BY p.name
    """, (id_battle,))
    team2 = into_dataframe(cur)
    
    return team1, team2

def get_teams_complete(cur: sqlite3.Cursor, game_id: int, _set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    team1, team2 = get_teams(cur,game_id, _set)
    missing_pkm = 6 - len(team2)
    avg_pok = get_avg_pokemon(cur)
    avg_pok["name"] = "avg"
    team2 = pd.concat([team2] + [avg_pok]*missing_pkm, ignore_index=True)
    return team1, team2

def get_avg_pokemon(cur: sqlite3.Cursor) -> pd.DataFrame:
    cur.execute("""
    SELECT avg(Pkm.base_hp) as base_hp, avg(Pkm.base_atk) as base_atk, avg(Pkm.base_def) as base_def, avg(Pkm.base_spa) as base_spa, avg(Pkm.base_spd) as base_spd, avg(Pkm.base_spe) as base_spe
    FROM Pokemon AS Pkm
    """)
    return into_dataframe(cur)

def get_team_pokemon_avg_pd(team: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(team.drop(columns=["name"], errors='ignore').mean()).T

def get_team_pokemon_avg(cur: sqlite3.Cursor, team_id: int) -> pd.DataFrame:

    cur.execute("""
    SELECT avg(Pkm.base_hp) as base_hp, avg(Pkm.base_atk) as base_atk, avg(Pkm.base_def) as base_def, avg(Pkm.base_spa) as base_spa, avg(Pkm.base_spd) as base_spd, avg(Pkm.base_spe) as base_spe
    FROM Pokemon AS Pkm, Level, Team
    WHERE Team.id = ? AND Level.team = Team.id AND Pkm.name = Level.pokemon
    """, (team_id,))

    return into_dataframe(cur)

def check_status_pokemon(cur: sqlite3.Cursor, team1: pd.DataFrame, team2: pd.DataFrame, id_battle: int) -> pd.DataFrame:
    cur.execute(
    """
        SELECT ps.*, p.*
        FROM Turn as t, PokemonState as ps, Pokemon as p
        WHERE t.battle = ? and t.p1_state = ps.id and p.name = ps.pokemon and 
        ORDER BY t.id asc
    """, (id_battle,)
    )
    
    cur.fetchall()[-1]