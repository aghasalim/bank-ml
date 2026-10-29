"""Models x {with, without duration}: expanding-window CV on train, then one chronological test evaluation."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bankml.data import load, time_split, xy
from bankml.metrics import bootstrap_ci, score
from bankml.models import make

warnings.filterwarnings("ignore")
train, test = time_split(load())
rows = []
for drop in (True, False):
    Xtr, ytr = xy(train, drop); Xte, yte = xy(test, drop)
    for name in ["dummy", "logreg", "rf", "xgb", "lgbm"]:
        cv_t = [roc_auc_score(ytr[v], make(name, drop).fit(Xtr.iloc[t], ytr[t]).predict_proba(Xtr.iloc[v])[:, 1])
                for t, v in TimeSeriesSplit(4).split(Xtr)]
        cv_r = [roc_auc_score(ytr[v], make(name, drop).fit(Xtr.iloc[t], ytr[t]).predict_proba(Xtr.iloc[v])[:, 1])
                for t, v in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0).split(Xtr, ytr)]
        p = make(name, drop).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        s = score(yte, p)
        rows.append({"model": name, "duration": "used" if not drop else "dropped",
                     "cv_random_roc": f"{np.mean(cv_r):.3f}±{np.std(cv_r):.3f}", "cv_time_roc": f"{np.mean(cv_t):.3f}±{np.std(cv_t):.3f}",
                     "test_roc": round(s["roc_auc"], 3), "test_roc_ci": bootstrap_ci(yte, p, roc_auc_score), "test_pr_lift": round(s["pr_auc"] / s["prevalence"], 2),
                     "brier": round(s["brier"], 3), "ece": round(s["ece"], 3)})
        print(rows[-1], flush=True)
for r in rows: r['test_roc_ci'] = tuple(float(x) for x in r['test_roc_ci'])
df = pd.DataFrame(rows); df.to_csv("results/baselines.csv", index=False)
print(f"\ntrain prevalence {ytr.mean():.3f}  test prevalence {yte.mean():.3f}  n_train {len(ytr)} n_test {len(yte)}")
print(df.to_markdown(index=False))
