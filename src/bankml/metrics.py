import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def score(y, p) -> dict:
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "ece": ece(y, p), "prevalence": float(np.mean(y))}


def bootstrap_ci(y, p, fn=average_precision_score, n: int = 500, seed: int = 0):
    g = np.random.default_rng(seed)
    v = [fn(y[i], p[i]) for i in (g.integers(0, len(y), len(y)) for _ in range(n)) if y[i].min() != y[i].max()]
    return tuple(np.percentile(v, [2.5, 97.5]).round(3))
