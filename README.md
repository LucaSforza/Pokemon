# Pokémon Battle Predictor — README

Brief description
-----------------
Project to predict the outcome of Pokémon battles from the early phases of a match. Raw data (jsonl files) are loaded into a SQLite DB, preprocessed, and used to train/evaluate various ML models.

What load_data.py does
----------------------
1. Creates a DB connection and registers a new `Dataset` (Train or Test).
2. For each match in the JSONL:
   - Inserts match metadata via [`insert_battle`](load_data.py).
   - Inserts team Pokémon with [`load_pokemon`](load_data.py) and [`load_team`](load_data.py).
   - Inserts turns with [`insert_turn`](load_data.py), which in turn calls [`insert_state_move`](load_data.py) to save states and moves.
3. Maintains consistency with `INSERT OR IGNORE` for reference tables (type, status, moves).
4. Final commit of the data into the DB.

data_analyzer folder
--------------------
The [data_analyzer/](data_analyzer/) folder contains code to extract, preprocess, select models and run experiments:

- [data_analyzer/__init__.py](data_analyzer/__init__.py) — exports main functions from `lib.py`.
- [data_analyzer/lib.py](data_analyzer/lib.py) — utilities and preprocessing / I/O pipeline:
  - [`get_datapoints`](data_analyzer/lib.py) — builds the dataset with all normalized features (as described in the report).
  - [`load_datapoints`](data_analyzer/lib.py) — reads preprocessed tables (`Input`, `Output`, `TestInput`, `TestOutput`).
  - [`create_submission`](data_analyzer/lib.py) — generates submission CSVs from predictions.
  - [`load_best_model`](data_analyzer/lib.py) — reconstructs a model from the information in `models.json`.
- [data_analyzer/model_selection.py](data_analyzer/model_selection.py) — classes and helpers for hyperparameter search and validation:
  - [`ModelTrainer`](data_analyzer/model_selection.py) and various implementations (LogisticRegressionTrainer, RandomForestClassifierTrainer, XGBClassifierTrainer, etc.) for cross‑validation and hyperparameter search.
  - [`plot_history`](data_analyzer/model_selection.py) — saves validation plots.
- [data_analyzer/__main__.py](data_analyzer/__main__.py) — CLI for analysis operations (save dataset, PCA, training, ensemble, etc.). Runs routines that use functions from `lib.py` and `model_selection.py`.

Typical execution
-----------------
- [main.py](main.py) contains the entire pipeline to reproduce the results; it was later converted into a notebook for the Kaggle challenge.

Useful resources in the repo
---------------------------
- Main script: [main.py](main.py)
- DB creation: [analisi/create_db.sql](analisi/create_db.sql)
- LaTeX report template: [Latex/](Latex/)
- Saved models info: [models.json](models.json)
- PCA importance/features: [pca.json](pca.json)
