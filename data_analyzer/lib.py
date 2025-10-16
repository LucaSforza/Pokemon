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
    SELECT b.id,p.name,p.base_hp,p.base_atk, p.base_def, p.base_spa,p.base_spd, p.base_spe
    FROM Level as l,Pokemon as p, Battle as b, Team as t
    WHERE t.id = b.team and l.team = t.id and l.pokemon = p.name and b.id = ?
        """, (id_battle,))
    team1 = into_dataframe(cur)
    
    cur.execute("""
    SELECT t.battle as id, p.name,p.base_hp,p.base_atk, p.base_def, p.base_spa,p.base_spd, p.base_spe 
    FROM Turn as t, PokemonState as ps, Pokemon as p, Battle as b
    WHERE t.battle = ? and b.id = t.battle and t.p2_state = ps.id and (ps.pokemon = p.name or b.p2_lead_pokemon = p.name)
    GROUP BY p.name
    """, (id_battle,))
    team2 = into_dataframe(cur)
    
    return team1, team2

    # TODO: add p2_lead_pokemon


def get_team_pokemon_avg(cur: sqlite3.Cursor, team_id: int) -> pd.DataFrame:

    cur.execute("""
    SELECT Pkm.name, Pkm.base_hp, Pkm.base_atk, Pkm.base_def, Pkm.base_spa, Pkm.base_spd, Pkm.base_spe 
    FROM Pokemon AS Pkm, Level, Team
    WHERE Team.id = ? AND Level.team = Team.id AND Pkm.name = Level.pokemon
    """, (team_id,))

    rows = cur.fetchall()
    pokemonStat = []

    for row in rows:
        name, p_hp, p_atk, p_def, p_spa, p_spd, p_spe = row
        pokemonStat.append((name, p_hp, p_atk, p_def, p_spa, p_spd, p_spe))
        print(f"- {name:<12} | HP:{p_hp:>3} | ATK:{p_atk:>3} | DEF:{p_def:>3} | SPA:{p_spa:>3} | SPD:{p_spd:>3} | SPE:{p_spe:>3}")
    
    avg_stats = [sum(stats[i] for stats in pokemonStat) / len(pokemonStat) for i in range(1, 7)]
    print(f"\nMedia Statistiche Team:")
    print(f"HP: {avg_stats[0]:.2f} | ATK: {avg_stats[1]:.2f} | DEF: {avg_stats[2]:.2f} | SPA: {avg_stats[3]:.2f} | SPD: {avg_stats[4]:.2f} | SPE: {avg_stats[5]:.2f}")
    
    # Creazione DataFrame con le medie
    df = pd.DataFrame({
        "Stat": ["HP", "ATK", "DEF", "SPA", "SPD", "SPE"],
        "Average": avg_stats
    })

    # Aggiungo una riga finale con la media complessiva
    df.loc[len(df.index)] = ["MEAN", sum(avg_stats) / len(avg_stats)]

    return df
