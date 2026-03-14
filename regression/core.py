import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def clean_xy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}

    if "x" not in cols or "y" not in cols:
        raise ValueError("Dataset must contain columns named 'x' and 'y'.")

    df = df.rename(columns={cols["x"]: "x", cols["y"]: "y"})[["x", "y"]]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("At least two valid observations are required.")

    return df


def regression_stats(df: pd.DataFrame) -> dict:
    df = clean_xy(df)

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    n = len(df)
    x_bar = x.mean()
    y_bar = y.mean()

    xy = x * y
    x2 = x**2
    y2 = y**2

    x_dev = x - x_bar
    y_dev = y - y_bar

    ss_xy = np.sum(x_dev * y_dev)
    ss_xx = np.sum(x_dev**2)
    ss_yy = np.sum(y_dev**2)

    b = ss_xy / ss_xx
    a = y_bar - b * x_bar

    y_hat = a + b * x
    residuals = y - y_hat

    r, _ = pearsonr(x, y)
    r2 = r**2

    return {
        "df": df,
        "n": n,
        "x_bar": x_bar,
        "y_bar": y_bar,
        "sum_x": x.sum(),
        "sum_y": y.sum(),
        "sum_xy": xy.sum(),
        "sum_x2": x2.sum(),
        "sum_y2": y2.sum(),
        "ss_xy": ss_xy,
        "ss_xx": ss_xx,
        "ss_yy": ss_yy,
        "a": a,
        "b": b,
        "r": r,
        "r2": r2,
        "y_hat": y_hat,
        "residuals": residuals,
    }


def worked_table(df: pd.DataFrame) -> pd.DataFrame:
    stats = regression_stats(df)
    out = stats["df"].copy()

    out["xy"] = out["x"] * out["y"]
    out["x²"] = out["x"] ** 2
    out["y²"] = out["y"] ** 2
    out["x̄"] = stats["x_bar"]
    out["ȳ"] = stats["y_bar"]
    out["x - x̄"] = out["x"] - stats["x_bar"]
    out["y - ȳ"] = out["y"] - stats["y_bar"]
    out["(x - x̄)(y - ȳ)"] = out["x - x̄"] * out["y - ȳ"]
    out["(x - x̄)²"] = out["x - x̄"] ** 2
    out["ŷ = a + bx"] = stats["a"] + stats["b"] * out["x"]
    out["Residual"] = out["y"] - out["ŷ = a + bx"]

    return out
