"""Load, clean and aggregate the Online Retail II transaction file.

Cleaning rules are derived from EDA in notebooks/01_eda.ipynb.
Each rule has an observed reason, recorded in the docstrings below.
"""

import pandas as pd

PRODUCT_CODE = r"^\d{5}[A-Z]*$"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def normalise_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Strip padding and force uppercase.

    EDA found '47503J ' (trailing space) and 172 codes existing in both
    cases, e.g. 85114B / 85114b, which split one product into two series.
    """
    out = df.copy()
    out["StockCode"] = out["StockCode"].astype(str).str.strip().str.upper()
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows representing real outbound demand for a stocked item."""
    out = normalise_codes(df).drop_duplicates()

    out = out[
        out["StockCode"].str.fullmatch(PRODUCT_CODE)
        & (out["Quantity"] > 0)
        & (out["Price"] > 0)
    ].copy()

    out["line_value"] = out["Quantity"] * out["Price"]
    return out


def select_top_skus(df: pd.DataFrame, n: int) -> list:
    """Top n SKUs by cleaned sales value. Must run AFTER clean()."""
    value = df.groupby("StockCode")["line_value"].sum().sort_values(ascending=False)
    return value.head(n).index.tolist()


def to_weekly(df: pd.DataFrame, skus: list) -> pd.DataFrame:
    """SKU x week demand table with silent weeks filled as zero.

    The zeros are the signal for ADI. A groupby alone would omit them.
    """
    sub = df[df["StockCode"].isin(skus)].copy()
    sub["week"] = sub["InvoiceDate"].dt.to_period("W").dt.start_time

    weekly = (sub.groupby(["StockCode", "week"])["Quantity"]
                 .sum()
                 .reset_index()
                 .rename(columns={"Quantity": "demand"}))

    all_weeks = pd.date_range(sub["week"].min(), sub["week"].max(), freq="W-MON")
    grid = pd.MultiIndex.from_product([skus, all_weeks], names=["StockCode", "week"])

    return (weekly.set_index(["StockCode", "week"])
                  .reindex(grid, fill_value=0)
                  .reset_index())


def build(config: dict) -> pd.DataFrame:
    raw = load_raw(config["data"]["raw_path"])
    print(f"Raw rows            : {len(raw):,}")

    cleaned = clean(raw)
    print(f"After cleaning      : {len(cleaned):,} rows "
          f"({len(cleaned)/len(raw):.1%} kept), "
          f"{cleaned['StockCode'].nunique():,} SKUs")

    skus = select_top_skus(cleaned, config["scope"]["top_n_skus"])
    share = (cleaned[cleaned["StockCode"].isin(skus)]["line_value"].sum()
             / cleaned["line_value"].sum())
    print(f"Scope               : top {len(skus)} SKUs = {share:.1%} of value")

    weekly = to_weekly(cleaned, skus)
    print(f"Weekly table        : {weekly.shape[0]:,} rows "
          f"({weekly['StockCode'].nunique()} SKUs x {weekly['week'].nunique()} weeks)")
    print(f"Zero-demand weeks   : {(weekly['demand'] == 0).mean():.1%}")

    weekly.to_parquet(config["data"]["weekly_path"], index=False)
    print(f"Saved               : {config['data']['weekly_path']}")
    return weekly