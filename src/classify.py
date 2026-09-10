"""Split products into fast and slow movers by how often they sell."""

import pandas as pd

ZERO_LIMIT = 0.30   # more than 30% empty weeks = slow mover


def classify(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per product: average demand, % of empty weeks, group."""
    summary = weekly.groupby("StockCode").agg(
        avg_demand=("demand", "mean"),
        total_demand=("demand", "sum"),
        zero_weeks=("demand", lambda s: (s == 0).mean()),
    ).reset_index()

    summary["group"] = summary["zero_weeks"].apply(
        lambda z: "slow" if z > ZERO_LIMIT else "fast"
    )
    return summary