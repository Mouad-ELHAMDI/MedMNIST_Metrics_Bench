from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except Exception:
    _XGB_OK = False
    XGBClassifier = None  # type: ignore

def make_classifier(kind: str):
    kind = kind.lower()
    if kind == "svm":
        return SVC()
    elif kind == "knn":
        return KNeighborsClassifier(n_jobs=-1)
    elif kind == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=0)
    elif kind == "xgboost":
        if not _XGB_OK:
            raise ImportError("xgboost is not installed. Install requirements-optional.txt")
        return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=0)
    else:
        raise ValueError(f"Unknown classifier: {kind}")

class NestedCVClassifier:
    def __init__(self, classifier_type: str, outer_splits: int = 3, inner_splits: int = 3, seed: int = 42):
        self.classifier_type = classifier_type
        self.outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
        self.inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    def perform(self, X: np.ndarray, y: np.ndarray, grid: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[float]]:
        model = make_classifier(self.classifier_type)
        outer_scores, best_params_list = [], []
        for tr, te in self.outer_cv.split(X):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            search = GridSearchCV(model, grid, cv=self.inner_cv, scoring="accuracy", n_jobs=-1)
            search.fit(X_tr, y_tr)
            best_params_list.append(search.best_params_)
            best = search.best_estimator_
            outer_scores.append(best.score(X_te, y_te))
        return best_params_list, outer_scores
