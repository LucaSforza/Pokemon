import json
import sqlite3
from typing import Any
import numpy as np
import pandas as pd

from copy import deepcopy


import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

#from IPython.display import display
from joblib import parallel_backend
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", category=UserWarning)

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

def into_dataframe_last(cur: sqlite3.Cursor) -> pd.DataFrame:
    row = cur.fetchall()[-1]
    cols = [desc[0] for desc in cur.description]
    return pd.DataFrame([row], columns=cols)

def get_pokemons(cur: sqlite3.Cursor) -> list[Pokemon]:
    cur.execute("SELECT * FROM Pokemon")
    return into_dataframe(cur)

def get_teams(cur: sqlite3.Cursor, game_id: int, _set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cur.execute("SELECT id FROM Dataset WHERE type = ?", (_set,))
    id_dataset = cur.fetchone()[0]
    cur.execute("SELECT b.id FROM bat_dat, Battle as b WHERE dataset = ? and b.id = bat_dat.battle and b.battle_id = ?", (id_dataset,game_id))
    id_battle = cur.fetchone()[0]
    
    cur.execute("""
    SELECT p.*
    FROM TeamP1 as tp,Pokemon as p, Battle as b
    WHERE tp.battle = b.id and tp.pokemon = p.name and b.id = ?
        """, (id_battle,))
    team1 = into_dataframe(cur)
    
    cur.execute("""
    SELECT p.*
    FROM Turn as t, PokemonState as ps, Pokemon as p, Battle as b
    WHERE t.battle = ? and b.id = t.battle and t.p2_state = ps.id and (ps.pokemon = p.name or b.p2_lead_pokemon = p.name)
    GROUP BY p.name
    """, (id_battle,))
    team2 = into_dataframe(cur)
    
    team1, team2 = get_team_with_types(team1, team2, cur, game_id, _set)

    return team1, team2

def get_team_with_types(team1: pd.DataFrame, team2: pd.DataFrame, cur: sqlite3.Cursor, game_id: int, _set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    team1_pokemons = tuple(team1['name'].tolist())
    team2_pokemons = tuple(team2['name'].tolist())
    
    cur.execute(f"""
        SELECT p.name, GROUP_CONCAT(tp.type, ', ') AS types
        FROM Pokemon AS p, type_pok AS tp
        WHERE p.name IN ({','.join('?'*len(team1_pokemons))}) and p.name = tp.pokemon
        GROUP BY p.name
    """, team1_pokemons)
    team1_types = into_dataframe(cur)

    cur.execute(f"""
        SELECT p.name, GROUP_CONCAT(tp.type, ', ') AS types
        FROM Pokemon AS p, type_pok AS tp
        WHERE p.name IN ({','.join('?'*len(team2_pokemons))}) and p.name = tp.pokemon
        GROUP BY p.name
    """, team2_pokemons)
    team2_types = into_dataframe(cur)

    team1_types['types'] = team1_types['types'].apply(lambda x: [t.strip() for t in x.split(',')])
    team2_types['types'] = team2_types['types'].apply(lambda x: [t.strip() for t in x.split(',')])
    
    team1_with_types = pd.merge(team1, team1_types, on="name", how="left")
    team2_with_types = pd.merge(team2, team2_types, on="name", how="left")

    return team1_with_types, team2_with_types


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

def get_team_pokemon_avg(cur: sqlite3.Cursor, id_battle: int) -> pd.DataFrame:

    cur.execute("""
    SELECT avg(Pkm.base_hp) as base_hp, avg(Pkm.base_atk) as base_atk, avg(Pkm.base_def) as base_def, avg(Pkm.base_spa) as base_spa, avg(Pkm.base_spd) as base_spd, avg(Pkm.base_spe) as base_spe
    FROM Pokemon AS Pkm, TeamP1 as tp
    WHERE tp.id = ? AND Pkm.name = tp.pokemon
    """, (id_battle,))

    return into_dataframe(cur)

def check_status_pokemon(cur: sqlite3.Cursor, team: pd.DataFrame, primary: bool,id_battle: int) -> pd.DataFrame:
    
    results = []
    
    string = "p1_state" if primary else "p2_state"
    
    for name in [team.iloc[i]["name"] for i in range(len(team))]:
        cur.execute(
        f"""
            SELECT MAX(t.id) AS turn, ps.*
            FROM Turn as t, PokemonState as ps
            WHERE t.battle = ? and ps.pokemon = ? and t.{string} = ps.id
        """,
        (id_battle, name)
        )
        frame = into_dataframe(cur)
        if not frame["turn"].isna().all():
            results.append(frame)
    return pd.concat(results, ignore_index=True)

def check_status_complete(cur: sqlite3.Cursor, team: pd.DataFrame, primary: bool,battle_id: int, _set: str) -> pd.DataFrame:
    team_complete = get_teams_complete(cur, battle_id, _set)[0 if primary else 1] 

    id_battle = find_id_battle(cur, battle_id, _set)
    pokemon_status = check_status_pokemon(cur, team, primary,id_battle)
    pokemon_status = pokemon_status.drop(columns=["turn", "id"])
    pokemon_status = pokemon_status.rename(columns={"pokemon": "name"})
    
    complete_pokemon_status = pd.merge(team_complete, pokemon_status, on="name", how="left")

    complete_pokemon_status = complete_pokemon_status.fillna({
        "hp_pct": 1, "boost_atk": 0, "boost_def": 0, "boost_spa": 0, "boost_spd": 0, "boost_spe": 0, "status": "nostatus", "pok_move": "nomove"})

    complete_pokemon_status["hp_final"] = complete_pokemon_status["base_hp"] * complete_pokemon_status["hp_pct"]
    complete_pokemon_status = complete_pokemon_status.drop(columns=["hp_pct", "base_hp", "pok_move"])

    columns_to_reset = ["base_atk", "base_def", "base_spa", "base_spd", "base_spe", "boost_atk", "boost_def", "boost_spa", "boost_spd", "boost_spe"]
    complete_pokemon_status.loc[complete_pokemon_status['hp_final'] == 0, columns_to_reset] = 0
    complete_pokemon_status.loc[complete_pokemon_status['hp_final'] == 0, ["types"]] = "fnt"

    complete_pokemon_status['types'] = complete_pokemon_status['types'].apply(
        lambda x: x if isinstance(x, (list)) else [x])
    
    return complete_pokemon_status

def find_id_battle(cur: sqlite3.Cursor, battle_id: int, _set: str) -> int:
    cur.execute("SELECT id FROM Dataset WHERE type = ?", (_set,))
    id_dataset = cur.fetchone()[0]
    cur.execute("SELECT b.id FROM bat_dat, Battle as b WHERE dataset = ? and b.id = bat_dat.battle and b.battle_id = ?", (id_dataset,battle_id))
    id_battle = cur.fetchone()[0]
    return id_battle

def print_battle(cur: sqlite3.Cursor, id_battle: int) -> None:
    cur.execute(
        """
        SELECT t.id as turn, ps.*
        FROM Turn as t, PokemonState as ps
        WHERE t.battle = ? and t.p1_state = ps.id
        """, (id_battle,)
    )
    
    print(into_dataframe(cur))
    
    cur.execute(
        """
        SELECT t.id as turn, ps.*
        FROM Turn as t, PokemonState as ps
        WHERE t.battle = ? and t.p2_state = ps.id
        """, (id_battle,)
    )
    
    print(into_dataframe(cur))

def get_all_types(cur: sqlite3.Cursor) -> list[str]:
    cur.execute("SELECT name FROM PokemonType")
    types = [row[0] for row in cur.fetchall()]
    return types

def get_all_status(cur: sqlite3.Cursor) -> list[str]:
    cur.execute("SELECT name FROM Status")
    status = [row[0] for row in cur.fetchall()]
    return status

# TODO: implement
# Fai una query alla tabella input e dividili in due datafram X e Y
def get_datapoints(cur: sqlite3.Cursor, _set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cur.execute("SELECT count(*) FROM Dataset as d, bat_dat bd WHERE d.type = ? and bd.dataset = d.id", (_set,))
    n_points = into_dict(cur)[0]["count(*)"]
    print(n_points)
    X = pd.DataFrame()
    Y = pd.DataFrame()
    
    for id in tqdm(range(n_points), desc="datapoints"):
        X, Y = get_teams_features(cur, id, _set,X, Y)

    return X, Y
       
    
# Ritornare un dataframe e una serie con 0 o 1
def get_teams_features(
    cur: sqlite3.Cursor, 
    battle_id: int, 
    _set: str, 
    all_status: pd.DataFrame, 
    all_results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    all_types = get_all_types(cur) + ["nan", "fnt"]

    mlb = MultiLabelBinarizer(classes=all_types)
    mlb.fit([all_types])

    all_pokemon_status = get_all_status(cur)
    ohe = OneHotEncoder(categories=[all_pokemon_status], sparse_output=False, handle_unknown='ignore')
    
    team1, team2 = get_teams(cur, battle_id, _set)

    status1 = check_status_complete(cur, team1,True, battle_id, _set)
    status2 = check_status_complete(cur, team2,False, battle_id, _set)              

    status1 = status1.drop(columns=["name"])
    status2 = status2.drop(columns=["name"])

    p1_type_encoded = pd.DataFrame(
        mlb.transform(status1['types']),
        columns=[f'type_{t}' for t in mlb.classes_],
        index=status1.index)

    p2_type_encoded = pd.DataFrame(                    
        mlb.transform(status2['types']),
        columns=[f'type_{t}' for t in mlb.classes_],
        index=status2.index)
    
    # attacco gli altri attributi, si riferiscono per ogni tabella e per ogni pokemon
    status1 = pd.concat([status1.drop(columns=['types']), p1_type_encoded], axis=1)
    status2 = pd.concat([status2.drop(columns=['types']), p2_type_encoded], axis=1)
    
    status1 = status1.drop(columns=[f'type_fnt'])
    status2 = status2.drop(columns=[f'type_fnt'])
    
    p1_status_encoded = pd.DataFrame(
        ohe.fit_transform(status1[['status']]),
        columns=[f'status_{s}' for s in all_pokemon_status],
        index=status1.index
    )

    p2_status_encoded = pd.DataFrame(
        ohe.fit_transform(status2[['status']]),
        columns=[f'status_{s}' for s in all_pokemon_status],
        index=status2.index
    )

    status1 = pd.concat([status1.drop(columns=['status']), p1_status_encoded], axis=1)
    status2 = pd.concat([status2.drop(columns=['status']), p2_status_encoded], axis=1)

    status_aggregated1 = pd.DataFrame(status1.sum()).T
    status_aggregated2 = pd.DataFrame(status2.sum()).T

    delta_status = status_aggregated1 - status_aggregated2
    id_battle = find_id_battle(cur, battle_id, _set)
    delta_status["id_battle"] = id_battle
    
    all_status = pd.concat([all_status, delta_status], ignore_index=True)

    result = get_battle_result(cur, battle_id, _set)
    result = pd.DataFrame([result], columns=["result"])
    result["id_battle"] = id_battle

    all_results = pd.concat([all_results, result], ignore_index=True)

    return all_status, all_results

def get_battle_result(cur: sqlite3.Cursor, battle_id: int, _set: str) -> int:
    id_battle = find_id_battle(cur, battle_id, _set)
    cur.execute("SELECT result FROM Battle WHERE id = ?", (id_battle,))
    result = cur.fetchone()[0]
    return result

def save_datapoints(conn: sqlite3.Connection, X: pd.DataFrame, Y: pd.DataFrame, test: bool = False) -> None:
    X_table = 'TestInput' if test else 'Input'
    Y_table = 'TestOutput' if test else 'Output'
    X.to_sql(X_table, conn, if_exists='replace', index=False)
    Y.to_sql(Y_table, conn, if_exists='replace', index=False)

def load_datapoints(conn: sqlite3.Connection, test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_table = 'TestInput' if test else 'Input'
    Y_table = 'TestOutput' if test else 'Output'
    X = pd.read_sql(f'SELECT * FROM {X_table}', conn)
    Y = pd.read_sql(f'SELECT * FROM {Y_table}', conn)

    return X, Y

def train(X: pd.DataFrame,Y: pd.DataFrame, seed: int=42, n_jobs=8):
    
    rng = np.random.default_rng(seed)
    split_seed = rng.integers(0, 2**32 - 1)
    model_seed = rng.integers(0, 2**32 - 1)
    
    regressor = LogisticRegressionCV(cv=5,n_jobs=n_jobs, max_iter=10000, random_state=model_seed)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=split_seed)
    
    regressor.fit(X_train,Y_train)
    Y_pred = regressor.predict(X_val)
    accuracy: float = accuracy_score(Y_val, Y_pred)
    
    y_pred_proba = regressor.predict_proba(X_val)[:,1]  # probabilità di classe 1
    r2 = r2_score(Y_val, y_pred_proba)
    return regressor, accuracy, r2

def train_to_submit(X: pd.DataFrame,Y: pd.DataFrame, seed: int=42, n_jobs=8):    
    regressor = LogisticRegressionCV(cv=5,n_jobs=n_jobs, max_iter=10000, random_state=seed)
    regressor.fit(X,Y)
    return regressor

def create_submission(cur: sqlite3.Cursor, model: Any, X_test: pd.DataFrame) -> None:
    #TODO caricare i dati di test e fare le predizioni
    # Make predictions on the test data
    test_df = find_test_id(cur)

    print("Generating predictions on the test set...")
    test_predictions = model.predict(X_test)
    print(f"Predictions generated. {test_predictions}")
    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })

    # Save the DataFrame to a .csv file
    submission_df.to_csv('submission.csv', index=False)

    print("\n'submission.csv' file created successfully!")
    #display(submission_df.head())

def scale_input(X: pd.DataFrame) -> np.ndarray:
    X = X.drop(columns=["id_battle"], errors='ignore')
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def find_test_id(cur: sqlite3.Cursor) -> int:
    cur.execute("SELECT count(*) FROM Dataset as d, bat_dat bd WHERE d.type = 'Test' and bd.dataset = d.id")
    n_points = into_dict(cur)[0]["count(*)"]

    df = pd.DataFrame({
        "battle_id": np.arange(5000)  # genera numeri da 0 a 4999
    })

    return df


def prepare_data(conn: sqlite3.Connection, variance: float):
    X,Y = load_datapoints(conn)
    
    X = X.sort_values(by="id_battle")
    Y = Y.sort_values(by="id_battle")
    
    total_importance = None
    
    with open("pca.json", "r") as f:
        total_importance = json.load(f)
    
    # calcolo varianza cumulativa
    sorted_items = sorted(total_importance.items(), key=lambda x: x[1], reverse=True)
    values = np.array([v for _, v in sorted_items])
    cum = np.cumsum(values) / np.sum(values)

    idx = np.argmax(cum >= variance)
    selected = [f for f, _ in sorted_items[: idx + 1]]
    print(selected)
    X = X.filter(items=selected)

    Y = Y.drop(columns=["id_battle"]).to_numpy()
    X = scale_input(X) 
    return X,Y