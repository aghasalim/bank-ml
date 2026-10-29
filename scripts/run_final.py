"""Tune LightGBM (no duration), calibrate, segment errors, SHAP. Outputs to results/."""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bankml.data import load, time_split, xy
from bankml.metrics import bootstrap_ci, score
from bankml.models import make

warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
train, test = time_split(load())
Xtr, ytr = xy(train); Xte, yte = xy(test)

# 1. Tuning: only train data, expanding-window CV, ROC-AUC (prevalence-robust).
def objective(t):
    kw = dict(n_estimators=t.suggest_int("n_estimators", 100, 600), learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
              num_leaves=t.suggest_int("num_leaves", 4, 64, log=True), min_child_samples=t.suggest_int("min_child_samples", 10, 200, log=True),
              reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
    return np.mean([roc_auc_score(ytr[v], make("lgbm", True, **kw).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[v])[:, 1])
                    for a, v in TimeSeriesSplit(4).split(Xtr)])
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=25)
best = study.best_params
default = make("lgbm", True).fit(Xtr, ytr); tuned = make("lgbm", True, **best).fit(Xtr, ytr)
pd_, pt = default.predict_proba(Xte)[:, 1], tuned.predict_proba(Xte)[:, 1]
out = [f"best params: {best}", f"time-CV ROC tuned={study.best_value:.3f}",
       f"test ROC default={roc_auc_score(yte, pd_):.3f} {bootstrap_ci(yte, pd_, roc_auc_score)}  tuned={roc_auc_score(yte, pt):.3f} {bootstrap_ci(yte, pt, roc_auc_score)}"]

# 2. Calibration: fit on first 80% of train, calibrate on most recent 20% of train, evaluate on test.
cut = int(len(Xtr) * 0.8)
base = make("lgbm", True, **best).fit(Xtr.iloc[:cut], ytr[:cut])
curves = {"uncalibrated": base.predict_proba(Xte)[:, 1]}
for m in ("sigmoid", "isotonic"):
    cal = CalibratedClassifierCV(FrozenEstimator(base), method=m).fit(Xtr.iloc[cut:], ytr[cut:])
    curves[m] = cal.predict_proba(Xte)[:, 1]
out.append(f"calibration slice prevalence={ytr[cut:].mean():.3f}, test prevalence={yte.mean():.3f}")
fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "k--", label="perfect")
for k, p in curves.items():
    s = score(yte, p); out.append(f"{k:13s} brier={s['brier']:.3f} ece={s['ece']:.3f} mean_pred={p.mean():.3f}")
    fx, fy = calibration_curve(yte, p, n_bins=10, strategy="quantile"); ax.plot(fy, fx, marker="o", label=k)
ax.set_xlabel("mean predicted"); ax.set_ylabel("observed fraction positive"); ax.legend(); ax.set_title("Reliability on chronological test")
fig.savefig("results/calibration.png", dpi=130, bbox_inches="tight")

# 3. Segment analysis on test (tuned model).
te = test.copy(); te["p"] = pt; te["age_band"] = pd.cut(te.age, [0, 29, 44, 59, 120], labels=["<30", "30-44", "45-59", "60+"])
seg = []
for col in ("age_band", "contact", "job", "poutcome"):
    for v, g in te.groupby(col, observed=True):
        if len(g) >= 150 and g.y.nunique() == 2:
            seg.append({"segment": f"{col}={v}", "n": len(g), "prevalence": round(g.y.mean(), 3), "mean_pred": round(g.p.mean(), 3), "roc_auc": round(roc_auc_score(g.y, g.p), 3)})
seg = pd.DataFrame(seg).sort_values("roc_auc"); seg.to_csv("results/segments.csv", index=False)
out += ["", "Worst 6 / best 3 segments by ROC-AUC (n>=150):", seg.head(6).to_markdown(index=False), seg.tail(3).to_markdown(index=False)]

# 4. SHAP on the tuned model, 2000 test rows.
pre, est = tuned.named_steps["pre"], tuned.named_steps["est"]
S = pre.transform(Xte.sample(2000, random_state=0)); S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
sv = shap.TreeExplainer(est).shap_values(S); sv = sv[1] if isinstance(sv, list) else sv
imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
out += ["", "Top 10 mean|SHAP|:", imp.head(10).round(4).to_markdown()]
plt.figure(); shap.summary_plot(sv, S, feature_names=names, max_display=12, show=False); plt.savefig("results/shap_summary.png", dpi=130, bbox_inches="tight")
Path("results/final.txt").write_text("\n".join(out)); print("\n".join(out))
