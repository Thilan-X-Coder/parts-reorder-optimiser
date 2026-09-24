"""Demand forecast models over the lead time."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["lag_1", "lag_2", "lag_4", "avg_4", "avg_8",
            "std_8", "week_of_year", "month"]

COMMON = dict(n_estimators=400, random_state=42)


def train(train_df: pd.DataFrame, kind: str = "lgbm", alpha: float = 0.95):
    """Train one forecast model.

    kind = "lgbm"     -> point forecast, optimised for MAE
    kind = "quantile" -> the level demand stays below `alpha` of the time
    kind = "rf"       -> Random Forest point forecast (comparison only)
    """
    if kind == "lgbm":
        model = lgb.LGBMRegressor(
            objective="mae", learning_rate=0.05, num_leaves=31,
            min_child_samples=30, verbose=-1, **COMMON)

    elif kind == "quantile":
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha, learning_rate=0.05,
            num_leaves=31, min_child_samples=30, verbose=-1, **COMMON)

    elif kind == "rf":
        model = RandomForestRegressor(
            max_depth=10, min_samples_split=30, n_jobs=-1, **COMMON)

    else:
        raise ValueError(f"Unknown model kind: {kind}")

    model.fit(train_df[FEATURES], train_df["target"])
    return model


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """Demand cannot be negative, so clip at zero."""
    return np.clip(model.predict(df[FEATURES]), 0, None)


def importance(model) -> pd.DataFrame:
    return (pd.DataFrame({"feature": FEATURES,
                          "importance": model.feature_importances_})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))