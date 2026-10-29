from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parents[2] / "data/bank-full.csv"
CAT = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
NUM = ["age", "balance", "day", "campaign", "pdays", "previous"]


def load(path: Path = CSV) -> pd.DataFrame:
    d = pd.read_csv(path, sep=";")
    d["y"] = (d["y"] == "yes").astype(int)
    d["was_contacted"] = (d["pdays"] != -1).astype(int)  # pdays=-1 means "never contacted before"
    return d


def time_split(d: pd.DataFrame, test_frac: float = 0.2):
    """Rows are in chronological order (UCI docs), so the split is by position: train = earlier, test = later."""
    n = int(len(d) * (1 - test_frac))
    return d.iloc[:n].reset_index(drop=True), d.iloc[n:].reset_index(drop=True)


def xy(d: pd.DataFrame, drop_duration: bool = True):
    # `duration` is only known after the call ends, i.e. after the outcome: using it leaks the target.
    X = d.drop(columns=["y"] + (["duration"] if drop_duration else []))
    return X, d["y"].to_numpy()
