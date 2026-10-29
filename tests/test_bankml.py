import numpy as np
import pytest

from bankml.data import load, time_split, xy
from bankml.metrics import ece
from bankml.models import make


@pytest.fixture(scope="module")
def d():
    return load()


def test_data_shape_and_target(d):
    assert d.shape == (45211, 18)  # 17 raw columns + was_contacted
    assert set(d.y.unique()) == {0, 1} and 0.10 < d.y.mean() < 0.13
    assert d.isna().sum().sum() == 0


def test_time_split_is_chronological_and_disjoint(d):
    tr, te = time_split(d, 0.2)
    assert len(tr) + len(te) == len(d)
    assert tr.equals(d.iloc[: len(tr)].reset_index(drop=True))  # train is strictly earlier rows


def test_duration_dropped_by_default(d):
    X, _ = xy(d)
    assert "duration" not in X and "y" not in X
    assert "duration" in xy(d, drop_duration=False)[0]


@pytest.mark.parametrize("name", ["logreg", "rf", "xgb", "lgbm"])
def test_pipeline_smoke(d, name):
    s = d.sample(3000, random_state=0)
    X, y = xy(s)
    p = make(name).fit(X[:2000], y[:2000]).predict_proba(X[2000:])[:, 1]
    assert p.shape == (1000,) and ((0 <= p) & (p <= 1)).all()


def test_ece_perfect_and_worst():
    assert ece(np.array([0, 1] * 50), np.array([0.0, 1.0] * 50)) == pytest.approx(0)
    assert ece(np.zeros(100), np.ones(100) * 0.99) == pytest.approx(0.99)
