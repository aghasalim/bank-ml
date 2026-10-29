from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import CAT, NUM

SEED = 0


def make(name: str, drop_duration: bool = True, **params) -> Pipeline:
    """All preprocessing lives inside the Pipeline, so CV folds never see each other's statistics."""
    num = NUM + ["was_contacted"] + ([] if drop_duration else ["duration"])
    scale = StandardScaler() if name == "logreg" else "passthrough"
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT), ("num", scale, num)]
    )
    est = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "xgb": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=4),
        "lgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=4),
    }[name]
    est.set_params(**params)
    return Pipeline([("pre", pre), ("est", est)])
