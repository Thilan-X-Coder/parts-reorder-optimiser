"""Time-based train/test split with an embargo gap."""

import pandas as pd


def time_split(df: pd.DataFrame, train_end: str, gap_weeks: int):
    """Split by date. The gap prevents the training target from
    overlapping into the test period."""
    train_end = pd.Timestamp(train_end)
    test_start = train_end + pd.Timedelta(weeks=gap_weeks)

    train = df[df["week"] <= train_end].copy()
    test = df[df["week"] >= test_start].copy()

    return train, test