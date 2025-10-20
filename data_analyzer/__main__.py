import sys

from lib import *

PROGRAM_NAME = sys.argv[0]

def usage():
    print(f"{PROGRAM_NAME} <command> [args...]")
    print("    get_pokemons <outputfile> <datasets...> salva nel file di output tutti i pokemon che trovi")
    print("    get_teams <outputfile> <dataset> <game id> <pokemon_file> prendi le squadre che trovi di un certo id")

def main():
    try:
        command = sys.argv[1]
    except IndexError:
        print("[ERROR] no command")
        usage()
        exit(1)
    try:
        database_path = sys.argv[2]
    except IndexError:
        print("[ERROR] no db specified")
        usage()
        exit(1)
    if command == "get_pokemons":
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            pokemons = get_pokemons(conn.cursor())
            print(pokemons)
    elif command == "get_teams":
        try:
            game_id = int(sys.argv[3])
            _set = sys.argv[4]
            if _set not in ["Test", "Train"]:
                print(f"[ERROR] the set {_set} doesn't exists")
                usage()
                exit(1)
        except IndexError:
            print("[ERROR] not enough arguments")
            usage()
            exit(1)
        except ValueError:
            print("[ERROR] the game id is not an integer")
            usage()
            exit(1)
        team1,team2 = None, None
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            team1, team2 = get_teams_complete(cur, game_id, _set)
            print(f"Team1:\n{team1}")
            print(f"Team2:\n{team2}")
            
            print(f"Avg Team1:\n{get_team_pokemon_avg_pd(team1)}")
            print(f"Avg Team2:\n{get_team_pokemon_avg_pd(team2)}")
    elif command == "avg":
        id_battle = int(sys.argv[3])
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            avg = get_team_pokemon_avg(conn.cursor(), id_battle)
            print(avg)
            all_avg = get_avg_pokemon(conn.cursor())
            print(all_avg)
    elif command == "pokemon_state":
        id_battle = int(sys.argv[3])
        _set = sys.argv[4]
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            team1, team2 = get_teams(cur, id_battle, _set)
            print(f"Team1:\n{team1}")
            print(f"Team2:\n{team2}")
            id_battle = find_id_battle(cur, id_battle, _set)
            status1 = check_status_pokemon(cur, team1,True, id_battle)
            status2 = check_status_pokemon(cur, team2,False, id_battle)
            print(f"Status1:\n{status1}")
            print(f"Status2:\n{status2}")
            
    elif command == "print_battle":
        id_battle = int(sys.argv[3])
        _set = sys.argv[4]
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            id_battle = find_id_battle(cur, id_battle, _set)
            print(f"id_battle = {id_battle}")
            print_battle(cur, id_battle)
    
    elif command == "complete_pokemon_state":
        battle_id = int(sys.argv[3])
        _set = sys.argv[4]
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            team1, team2 = get_teams(cur, battle_id, _set)
            status1 = check_status_complete(cur, team1,True, battle_id, _set)
            status2 = check_status_complete(cur, team2,False, battle_id, _set)
            print(f"Status1:\n{status1}")
            print(f"Status2:\n{status2}")

    elif command == "pca":
        _set = sys.argv[3]
        all_status = pd.DataFrame()

        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
  
            for battle_id in tqdm(range(1000), desc="Processing"): # TODO: piu generico
                all_status = get_teams_features(cur, battle_id, _set, all_status)
            
            scaler = StandardScaler()
            all_status_scaled = scaler.fit_transform(all_status)

            pca = PCA(n_components=0.95)  # conserva il 95% della varianza
            pca_result = pca.fit_transform(all_status_scaled)

            explained = pd.Series(pca.explained_variance_ratio_, index=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))])

            components = pd.DataFrame(pca.components_, columns=all_status.columns)

            # Prendi la prima componente principale (PC1)
            pc1_importance = components.loc[0].abs().sort_values(ascending=False)

            # Plot
            plt.figure(figsize=(10,6))
            pc1_importance.plot(kind='bar')
            plt.title('Top colonne per la prima componente principale (PC1)')
            plt.ylabel('Contributo assoluto')
            plt.savefig("plt/Contr_Absol_Train.png")
        
            # Pondera i loadings per la varianza spiegata
            weighted_importance = components.T * explained.values
            total_importance = weighted_importance.abs().sum(axis=1).sort_values(ascending=False)
        
            # Plot
            plt.figure(figsize=(10,6))
            total_importance.plot(kind='bar')
            plt.title('Contributo totale delle colonne alla PCA')
            plt.ylabel('Contributo totale ponderato')
            plt.savefig("plt/Contr_Ponderate_Train.png")

            #Stampo dizionario con le componenti principali in ordine di importanza
            print("Importanza delle caratteristiche (colonne) nella PCA:")
            for feature, importance in total_importance.items():
                print(f"{feature}: {importance}")


    if command == "train":
        # da capire 'solver' cosa fa
        regressor = LogisticRegressionCV()
        
        with sqlite3.connect(database_path) as conn:
            cur = conn.cursor()
            X,Y = get_datapoints(cur, "Train")
        X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=42)
        
        n_epochs = int(sys.argv[3])
        
        for _epoch in range(n_epochs):
            regressor.fit(X_train,Y_train)
            Y_pred = regressor.predict(X_train)
            
    
    else:
        print(f"[ERROR] unknown command {command}")
        usage()
        exit(1)
    
with parallel_backend('threading', n_jobs=8):
    main()