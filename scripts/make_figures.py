"""README figures from data/bank-full.csv and results/baselines.csv. Run: python scripts/make_figures.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bankml.data import load  # noqa: E402

OUT = ROOT / "results"

# 1. Drift: positive rate over row order (rows are chronological), in 20 equal blocks.
d = load()
blocks = np.array_split(d["y"].to_numpy(), 20)
rates = [b.mean() for b in blocks]
cut = int(len(d) * 0.8)
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(20) * len(d) / 20 + len(d) / 40
ax.bar(x, rates, width=len(d) / 20 * 0.85, color="#4c72b0")
ax.axvline(cut, color="k", ls="--", lw=1)
ax.text(cut, max(rates) * 1.12, "  test (last 20%)", va="top")
ax.text(cut, max(rates) * 1.12, "train (first 80%)  ", va="top", ha="right")
ax.set_ylim(0, max(rates) * 1.15)
ax.set_xlabel("row index (chronological order)")
ax.set_ylabel("share subscribed (y = yes)")
ax.set_title(f"Positive rate by row block: train {d.y[:cut].mean():.1%}, test {d.y[cut:].mean():.1%}")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "drift.png", dpi=150)

# 2. Random-CV vs chronological test ROC-AUC, with and without the leaky `duration` feature.
b = pd.read_csv(OUT / "baselines.csv")
b = b[b.model != "dummy"]
b["cv"] = b.cv_random_roc.str.split("±").str[0].astype(float)
fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
for ax, (dur, g) in zip(axes, b.groupby("duration", sort=False)):
    i = np.arange(len(g))
    ax.bar(i - 0.2, g.cv, 0.4, label="random 5-fold CV (train)", color="#c0c0c0")
    ax.bar(i + 0.2, g.test_roc, 0.4, label="chronological test", color="#4c72b0")
    ax.set_xticks(i, g.model)
    ax.set_title(f"duration {dur}")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("ROC-AUC")
axes[0].set_ylim(0.4, 1.0)
axes[0].legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "baselines.png", dpi=150)
print("wrote results/drift.png results/baselines.png")
