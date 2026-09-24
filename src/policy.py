"""Simulate a weekly review inventory policy over the test period."""

import numpy as np
import pandas as pd


def simulate(df: pd.DataFrame, rop_col: str, lead_time: int,
             cover_weeks: int = 4) -> dict:
    """Walk each SKU week by week.

    Weekly review. When stock plus stock already on order falls to the
    reorder point, place an order. Orders arrive after `lead_time` weeks.
    """
    n_weeks = df["week"].nunique()

    total_demand = 0.0
    total_filled = 0.0
    unmet = 0.0
    stock_weeks = []
    stockout_weeks = 0

    for _, g in df.sort_values(["StockCode", "week"]).groupby("StockCode"):
        rop = g[rop_col].to_numpy()
        demand = g["demand"].to_numpy()
        weekly_avg = g["avg_4"].to_numpy()

        stock = rop[0]
        pipeline = {}

        for t in range(len(g)):
            stock += pipeline.pop(t, 0.0)

            filled = min(stock, demand[t])
            stock -= filled

            total_demand += demand[t]
            total_filled += filled
            unmet += demand[t] - filled
            if filled < demand[t]:
                stockout_weeks += 1
            stock_weeks.append(stock)

            on_order = sum(pipeline.values())
            if stock + on_order <= rop[t]:
                target = rop[t] + cover_weeks * weekly_avg[t]
                qty = max(target - stock - on_order, 0.0)
                pipeline[t + lead_time] = pipeline.get(t + lead_time, 0.0) + qty

    return {
        "fill_rate": total_filled / total_demand,
        "avg_stock_units": float(np.sum(stock_weeks)) / n_weeks,
        "stockout_weeks": stockout_weeks,
        "unmet_units": unmet,
        "total_weeks": n_weeks,
    }


def cost(result: dict, unit_cost: float, holding_rate: float,
         stockout_cost: float) -> dict:
    """Convert a simulation result into money over the test period."""
    years = result["total_weeks"] / 52
    holding = result["avg_stock_units"] * unit_cost * holding_rate * years
    shortage = result["unmet_units"] * stockout_cost
    return {"holding": holding, "shortage": shortage,
            "total": holding + shortage}