"""Baseline forecast: recent average, scaled to the lead time."""

import numpy as np
import pandas as pd


def predict(df: pd.DataFrame, lead_time: int) -> pd.Series:
    """Assume the next `lead_time` weeks look like the last 4 weeks."""
    return df["avg_4"] * lead_time


def score(actual: pd.Series, predicted: pd.Series) -> dict:
    error = predicted - actual
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt((error ** 2).mean())),
        "Bias": float(error.mean()),
        "Mean actual": float(actual.mean()),
    }