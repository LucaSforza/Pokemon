import sys

from lib import *
from model_selection import *

PROGRAM_NAME = sys.argv[0]

def usage():
    print(f"{PROGRAM_NAME} <command> [args...]")
    print("    get_pokemons <outputfile> <datasets...> salva nel file di output tutti i pokemon che trovi")
    print("    get_teams <outputfile> <dataset> <game id> <pokemon_file> prendi le squadre che trovi di un certo id")

def main():
    print(sys.argv)
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
        # _set = sys.argv[3]

        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            X, Y = load_datapoints(conn)
            
            X = X.drop(columns=["id_battle"])
            pca = PCA(n_components=0.95)  # conserva il 95% della varianza
            pca_result = pca.fit_transform(X)

            explained = pd.Series(pca.explained_variance_ratio_, index=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))])

            components = pd.DataFrame(pca.components_, columns=X.columns)

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
            with open("pca.json", "w") as f:
                f.write(json.dumps(total_importance.to_dict(), indent=2))

    elif command == "train":
        variance = None
        try:
            variance = float(sys.argv[3])
        except IndexError:
            variance = 0.999
        
        with sqlite3.connect(database_path) as conn:
            cur = conn.cursor()
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

        Y = Y.drop(columns=["id_battle"])
        X = scale_input(X)     
        
        np.random.seed(42)
        numbers = np.random.randint(0, 2**32, size=100, dtype=np.uint64)
        total_accuracy = 0.0
        total_bias = 0.0
        total_r2 = 0.0
        
        for seed in numbers:
            rng = np.random.default_rng(seed)
            split_seed = rng.integers(0, 2**32 - 1)
            model_seed = rng.integers(0, 2**32 - 1)
            trainer = RandomForestClassifier(criterion="entropy", max_depth=97, max_features='sqrt', random_state=model_seed)
            print("------------------")
            model, accuracy, r2 = train(X,Y,trainer, seed=split_seed)
            total_accuracy += accuracy
            # total_bias += model.
            total_r2 += r2
            print(f"seed: {seed}")
            print("model: ", model.get_params(deep=True))
            print("Accuracy: ",accuracy)
            # print("Coefficients: ", model.coef_)
            # print("Bias: ", model.intercept_)
            print("R2: ", r2)
        print(f"Mean accuracy: {total_accuracy/len(numbers)}")
        # print(f"Mean bias: {total_bias/len(numbers)}")
        print(f"Mean R2: {total_r2/len(numbers)}")
        
    elif command == "save_train_data":
        with sqlite3.connect(database_path) as conn:
            cur = conn.cursor()
            X,Y = get_datapoints(cur, "Train")
            save_datapoints(conn, X, Y, False)

    elif command == "save_test_data":
        with sqlite3.connect(database_path) as conn:
            cur = conn.cursor()
            X,Y = get_datapoints(cur, "Test")
            save_datapoints(conn, X, Y, True)

    elif command == "create_submission":
        variance = None
        try:
            variance = float(sys.argv[3])
        except IndexError:
            variance = 0.999
        
        with sqlite3.connect(database_path) as conn:
            cur = conn.cursor()
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

        Y = Y.drop(columns=["id_battle"])
        X = scale_input(X)    
        
        models_name = ["LogisticRegression", "KNN", "RandomForest", "XGBoost", "DecisionTree"]
        estimators = [(name, load_best_model(name)) for name in models_name]
        final_estimator = LogisticRegressionCV(cv=5, max_iter=10000, random_state=42)  # combines base model predictions

        model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5 
        )

        model.fit(X,Y)
        
        with sqlite3.connect(database_path) as conn:
            cur = conn.cursor()
            X_test, _ = load_datapoints(conn, test=True)
            X_test = X_test.sort_values(by="id_battle")
            X_test = X_test.filter(items=selected)
            X_test = scale_input(X_test)    

            create_submission(cur, model, X_test)

    elif command == "model_selection":
        X,Y = None,None
        with sqlite3.connect(database_path) as conn:
            X,Y = prepare_data(conn, 0.999)
        
        model = LogisticRegressionTrainer()

        best_model, acc, validations = model.model_selection(X,Y)
        print(f"Model:\n{json.dumps(best_model, indent=2)}")
        print(f"accuracy: {acc}")
        plot_history(validations, type(model).__name__)
        
    
    elif command == "ensable":
        output_file = sys.argv[2]
        files = sys.argv[3:]
        view_file = sys.argv[3]
        view_file = pd.read_csv(view_file)
        battles = view_file["battle_id"].max()
        models = [pd.read_csv(file) for file in files ]
        result = pd.DataFrame(columns=["battle_id","player_won","confidence"])
        print(result.columns)
        for id in tqdm(range(battles+1)):
            labels = {}
            for model in models:
                label = model.loc[model["battle_id"] == id, "player_won"].values[0]
                labels[label] = labels.get(label, 0) + 1
            result_label, favor = max(labels.items(), key=lambda x: x[1])
            new_entry = {"battle_id": id, "player_won": result_label, "confidence": favor}
            result.loc[len(result)] = new_entry
        result = result.drop("confidence",axis=1)
        result.to_csv(output_file, index=False)            
    
    elif command == "meta_model":
        # --- Base learners ---
        models_name = ["LogisticRegression", "KNN", "RandomForest", "XGBoost", "DecisionTree"]

        estimators = [(name, load_best_model(name)) for name in models_name]

        # --- Meta-learner ---
        final_estimator = LogisticRegressionCV(cv=5, max_iter=10000, random_state=42)

        # --- Define Stacking ensemble ---
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5 
        )

        # --- Load data ---
        with sqlite3.connect(database_path) as conn:
            X,Y = prepare_data(conn, 0.999)  

        # --- Train and evaluate ---
        stacking_clf.fit(X, Y)

        scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, valid_idx in kf.split(X, Y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = Y[train_idx], Y[valid_idx]
            
            # Fitta su parte train (che includerà anche i fold interni per il passo meta-learner)
            stacking_clf.fit(X_train, y_train)
            
            # Valuta su validation fold
            score = stacking_clf.score(X_valid, y_valid)
            scores.append(score)
            print(f"Fold validation score: {score:.4f}")

        print(f"Mean validation score: {np.mean(scores):.4f}")

    else:
        print(f"[ERROR] unknown command {command}")
        usage()
        exit(1)
    
with parallel_backend('threading', n_jobs=8):
    main()