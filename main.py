import sys 
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'data_analyzer'))

from lib import *
import importlib.util

def execute_command_data_analyzer():
    module_folder = "data_analyzer"
    main_file = os.path.join(module_folder, "__main__.py")

    spec = importlib.util.spec_from_file_location("data_analyzer_main", main_file)
    data_analyzer_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_analyzer_main)

    data_analyzer_main.main()


def create_database(db_name):    
    # Check if the database file exists
    if not os.path.exists(db_name):
        print(f"Creation of {db_name}...")

        with open("analisi/create_db.sql", "r", encoding="utf-8") as f:
            sql_script = f.read()

        with sqlite3.connect(db_name) as conn:
            conn.executescript(sql_script)

        print(f"Database {db_name} correctly created.")

        spec = importlib.util.spec_from_file_location("load_data", "load_data.py")
        load_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(load_data)
        
        sys.argv = ["load_data.py", db_name, "input/train.jsonl", "input/test.jsonl"]
        
        load_data.main()

        print(f"Data correctly uploaded.")

    else:
        print(f"Database {db_name} already exists.")

def data_elaboration(db_name):
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()

        # Check if the elaborated data (train and test) tables already exists and contains data
        for folder in ["test", "train"]:
            table = 'TestInput' if folder == "test" else 'Input'
            
            cursor.execute("""
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name=?;
            """, (table, ))
            table_exists = cursor.fetchone() is not None

            if table_exists:
                cursor.execute("SELECT COUNT(*) FROM TestInput;")
                count = cursor.fetchone()[0]
                print(count)
                if count > 0:
                    print(f"Elaborated data already loaded in the db {db_name}.")
                    return 
           
            print("Elaborating and loading data...")
            
            arg2 = "save_" + folder + "_data"
            sys.argv = ["data_analyzer", arg2, db_name]
            execute_command_data_analyzer()

        print(f"Elaborated data successfully loaded in the db {db_name}.")
                
def load_model(model_name):
    with open("models.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    model_trained = list(data.keys())
    
    if model_name in model_trained:
        model = load_best_model(model_name) 
        print(f"Model {model_name} successfully loaded")
        return model
    elif model_name == "MetaModel":
        models_name = ["LogisticRegression", "KNN", "RandomForest", "XGBoost", "DecisionTree"]
        estimators = [(name, load_best_model(name)) for name in models_name]
        final_estimator = LogisticRegressionCV(cv=5, max_iter=10000, random_state=42)
        
        print(f"Model {model_name} successfully loaded")
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5 
        )
    
    print(f"Please choose one of this models {model_trained + ["MetaModel"]}")
    return 

def result_predictions(db_name, model):
    variance = None
    try:
        variance = float(sys.argv[3])
    except IndexError:
        variance = 0.999
    
    with sqlite3.connect(db_name) as conn:
        cur = conn.cursor()
        X,Y = load_datapoints(conn)
    
    
    X = X.sort_values(by="id_battle")
    Y = Y.sort_values(by="id_battle")
    
    total_importance = None
    
    with open("pca.json", "r") as f:
        total_importance = json.load(f)
    
    sorted_items = sorted(total_importance.items(), key=lambda x: x[1], reverse=True)
    values = np.array([v for _, v in sorted_items])
    cum = np.cumsum(values) / np.sum(values)

    idx = np.argmax(cum >= variance)
    selected = [f for f, _ in sorted_items[: idx + 1]]
    print(selected)
    X = X.filter(items=selected)

    Y = Y.drop(columns=["id_battle"])
    X = scale_input(X)    

    model.fit(X,Y)
    
    with sqlite3.connect(db_name) as conn:
        cur = conn.cursor()
        X_test, _ = load_datapoints(conn, test=True)
        X_test = X_test.sort_values(by="id_battle")
        X_test = X_test.filter(items=selected)
        X_test = scale_input(X_test)    

        create_submission(cur, model, X_test)


# ========== PIPELINE STEPS ======================================================

db_name = "pokemon.db"

# Create db and upload data if it hasn't been done before
create_database(db_name)

# Load worked data if it hasn't been done before
data_elaboration(db_name)

# Choose model to use from (["LogisticRegression", "KNN", "RandomForest", "XGBoost", "DecisionTree", "Ensemble", "MetaModel"])
model_name = "Ensemble"

if model_name == "Ensemble":
    models_name = ["LogisticRegression", "KNN", "RandomForest", "XGBoost", "DecisionTree"]
    estimators = [(name, load_best_model(name)) for name in models_name]    
    
    for (name, estimator) in estimators:
        result_predictions(db_name, estimator)
    
    classifiers = ["LogisticRegressionCV", "KNeighborsClassifier", "RandomForest", "XGBClassifier", "DecisionTreeClassifier"]
    submission_file = ["plt/" + c + "-submission.csv" for c in classifiers]

    sys.argv = ["data_analyzer", "ensable", db_name, "plt/ensable-submission1.csv"] + submission_file
    execute_command_data_analyzer()

else:
    # Load best parameters for that model 
    model = load_model(model_name)

    # Check if the model was successfully loaded
    if model == None:
        exit(1)

    #Create the submission file
    result_predictions(db_name, model)

# check_differences("plt/LogisticRegressionCV-submission.csv", "plt/LogisticRegression-submission.csv")
