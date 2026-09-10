"""Lag and rolling-window features per product, plus the forecast target."""

import pandas as pd


def build_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per product-week, described only by earlier weeks.

    Every lag and window is shifted by one week so a row never sees its own
    demand -- that would leak the answer into the model.
    """
    out = weekly.sort_values(["StockCode", "week"]).copy()
    g = out.groupby("StockCode")["demand"]

    for lag in (1, 2, 4):
        out[f"lag_{lag}"] = g.shift(lag)

    past = g.shift(1)
    by_sku = past.groupby(out["StockCode"])

    out["avg_4"] = by_sku.rolling(4).mean().reset_index(level=0, drop=True)
    out["avg_8"] = by_sku.rolling(8).mean().reset_index(level=0, drop=True)
    out["std_8"] = by_sku.rolling(8).std().reset_index(level=0, drop=True)

    out["week_of_year"] = out["week"].dt.isocalendar().week.astype(int)
    out["month"] = out["week"].dt.month

    feature_cols = ["lag_1", "lag_2", "lag_4", "avg_4", "avg_8", "std_8"]
    return out.dropna(subset=feature_cols).reset_index(drop=True)


def add_target(df: pd.DataFrame, lead_time: int = 8) -> pd.DataFrame:
    """Target = total demand over the next `lead_time` weeks."""
    out = df.sort_values(["StockCode", "week"]).copy()
    g = out.groupby("StockCode")["demand"]

    future = sum(g.shift(-i) for i in range(1, lead_time + 1))
    out["target"] = future

    return out.dropna(subset=["target"]).reset_index(drop=True)
