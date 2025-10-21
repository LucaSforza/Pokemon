from lib import *

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from abc import ABC,abstractmethod

class Model(ABC):
    
    @abstractmethod
    def predict(X: np.ndarray) -> np.ndarray:
        ...
    
    @abstractmethod
    def get_model_type() -> type:
        ...

class ModelTrainer(ABC):
    
    @abstractmethod
    def fit(X: np.ndarray, Y:np.ndarray, cv=5, n_jobs=8, seed=42) -> tuple[Model,float]:
        ...


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
    
    def fit(X: np.ndarray, Y:np.ndarray, cv=5, n_jobs=8, seed=42) -> tuple[Model,float]:
        model = LogisticRegressionCV(cv=cv,random_state=seed, n_jobs=n_jobs)
        model.fit(X,Y)
        mean_acc = np.mean([v.mean() for v in model.scores_.values()])
        return model, mean_acc
    
class RidgeTrainer(ModelTrainer):
    
    def fit(X: np.ndarray, Y:np.ndarray, cv=5, n_jobs=8, seed=42) -> tuple[Model,float]:
        model = RidgeCV(cv=cv,random_state=seed, n_jobs=n_jobs)
        model.fit(X,Y)
        mean_acc = np.mean([v.mean() for v in model.scores_.values()])
        return model, mean_acc
    
class KNeighborsClassifierTrainer(ModelTrainer):
    
    def fit(X: np.ndarray, Y:np.ndarray, cv=5, n_jobs=8, seed=42) -> tuple[Model,float]:
        model = KNeighborsClassifier()
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,                 # 5-fold cross-validation
            n_jobs=n_jobs         # usa tutti i core disponibili
        )

        grid.fit(X, Y)
        mean_acc = grid.best_score_
        return grid.best_estimator_, mean_acc