from lib import *

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, StratifiedKFold

from abc import ABC,abstractmethod

class Model(ABC):
    
    @abstractmethod
    def predict(self,X: np.ndarray) -> np.ndarray:
        ...
    
class ModelTrainer(ABC):
    
    @abstractmethod
    def get_best_hyperparams(self, model: Any) -> dict[str, Any]:
        ...
    
    @abstractmethod
    def cross_validation(self, size: int, X: np.ndarray, Y: np.ndarray) -> tuple[Model, float]:
        ...
    
    def fit(self, X: np.ndarray, Y:np.ndarray, cv=5, n_jobs=8, seed=42, patience=100, epochs=1000) -> tuple[Model,float, list[float]]:
        self.seed = seed
        self.n_job = n_jobs
        self.cv = cv
        validations = []
        best_model = None
        best_accuracy = None
        no_improve = 0
        for size in tqdm(range(1,epochs+1)):
            model, mean_acc = self.cross_validation(size, X, Y)
            validations.append(1 - mean_acc)
            if best_accuracy is None or best_accuracy < mean_acc:
                best_accuracy = mean_acc
                best_model = model
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        return best_model, best_accuracy, validations


def model_selections(models: dict[str, ModelTrainer], X: np.ndarray, Y: np.ndarray,seed=42, n_jobs=8) -> dict[str, Any]:
    
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    best_models = {}
    for name, model in models.items():
        new_seed = rng.integers(0, 2**32 - 1)
        m, accuracy = model.fit(X,Y,n_jobs=n_jobs, seed=new_seed)
        best_models[name] = {"model": m, "seed": new_seed, "accuracy": accuracy}
    
    return best_models

class LogisticRegressionTrainer(ModelTrainer):
    
    def fit(self,X: np.ndarray, Y:np.ndarray, cv=5, n_jobs=8, seed=42) -> tuple[Model,float]:
        model = LogisticRegressionCV(cv=cv,random_state=seed, n_jobs=n_jobs)
        model.fit(X,Y)
        mean_acc = np.mean([v.mean() for v in model.scores_.values()])
        return model, mean_acc
    
class RidgeTrainer(ModelTrainer):
    
    def fit(self,X: np.ndarray, Y:np.ndarray, cv=5, n_jobs=8, seed=42) -> tuple[Model,float]:
        model = RidgeCV(cv=cv)
        model.fit(X,Y)
        mean_acc = np.mean([v.mean() for v in model.scores_.values()])
        return model, mean_acc
    
class KNeighborsClassifierTrainer(ModelTrainer):
    
    def get_best_hyperparams(self, model) -> dict[str, Any]:
        model: KNeighborsClassifier = model
        return model.get_params(deep=True)
    
    def cross_validation(self, size: int, X: np.ndarray, Y: np.ndarray):
        model = KNeighborsClassifier(
            n_neighbors = size
        )

        param_grid = {
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=self.cv,
            n_jobs=self.n_job
        )

        grid.fit(X, Y)
        mean_acc = grid.best_score_
        return self.get_best_hyperparams(grid.best_estimator_), mean_acc

# TODO: Convertire al nuovo sistema
class XGBClassifierTrainer(ModelTrainer):
    
    def fit(self,X: np.ndarray, Y: np.ndarray, cv=5, n_jobs=8, seed=42) -> tuple[Model, float]:
        model = XGBClassifier(
            objective="binary:logistic",
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss"
        )

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )

        grid.fit(X, Y)
        mean_acc = grid.best_score_
        return grid.best_estimator_, mean_acc
    
class RandomForestClassifierTrainer(ModelTrainer):
    
    def get_best_hyperparams(self, model) -> dict[str, Any]:
        model: RandomForestClassifier = model
        return model.get_params(deep=True)
    
    def cross_validation(self, size: int, X: np.ndarray, Y: np.ndarray) -> tuple[Model,float]:
        model = RandomForestClassifier(
            max_depth=size
        )

        param_grid = {
            "criterion": ['gini', 'entropy'],
            "max_features": ['sqrt', 'log2'],
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=self.cv,
            n_jobs=self.n_job
        )

        grid.fit(X, Y)
        mean_acc = grid.best_score_
        return self.get_best_hyperparams(grid.best_estimator_), mean_acc

# TODO: convertire al nuovo sistema 
# Comunque è sempre retro compatibile questo codice
class DecisionTreeClassifierTrainer(ModelTrainer):
    
    def fit(self,X: np.ndarray, Y:np.ndarray, cv=5, n_jobs=4, seed=42, patience=20, epochs=1000) -> tuple[Model,float]:
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        
        best_model = None
        best_accuracy = None
        no_improve = 0
        for size in tqdm(range(1,epochs), desc="decision tree"):
            tot_accuracy = 0.0
            i = 0
            model = None
            for train_idx, test_idx in kf.split(X):
                model = DecisionTreeClassifier(max_depth=size)
                i += 1
                X_train, X_val = X[train_idx], X[test_idx]
                Y_train, Y_val = Y[train_idx], Y[test_idx]
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_val)
                accuracy = accuracy_score(Y_val, Y_pred)
                tot_accuracy += accuracy
            mean_acc = tot_accuracy/i
            if best_accuracy is None or best_accuracy < mean_acc:
                best_accuracy = mean_acc
                best_model = model
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        return best_model, best_accuracy