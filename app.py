"""Weekly reorder list for the parts planning team."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

import src.artifacts as artifacts
import src.features as features

st.set_page_config(page_title="Parts Reorder Optimiser", layout="wide")

REQUIRED_COLUMNS = ["StockCode", "on_hand"]
BUNDLE_PATH = "models/bundle.pkl"

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)


@st.cache_resource
def load_model() -> dict:
    """Cached as a resource: one shared copy of the models, never re-pickled."""
    if not Path(BUNDLE_PATH).exists():
        st.error(f"No trained model at {BUNDLE_PATH}. Run `python run_train.py` first.")
        st.stop()
    return artifacts.load_bundle(BUNDLE_PATH)


@st.cache_data
def score_latest() -> pd.DataFrame:
    """Score each part's most recent week with the saved models.

    Features come from the same build_features used in training, so the two
    cannot drift apart. No add_target here: the target is the future and does
    not exist at serving time.
    """
    bundle = load_model()
    weekly = pd.read_parquet(cfg["data"]["weekly_path"])
    latest = (features.build_features(weekly)
              .sort_values("week").groupby("StockCode").tail(1).copy())

    missing = [c for c in bundle["features"] if c not in latest.columns]
    if missing:
        st.error(f"Model expects feature(s) not built by the pipeline: "
                 f"{', '.join(missing)}. Retrain with `python run_train.py`.")
        st.stop()

    X = latest[bundle["features"]]
    latest["pred_lgbm"] = np.clip(bundle["point_model"].predict(X), 0, None)
    latest["reorder_point"] = np.clip(bundle["quantile_model"].predict(X), 0, None)

    groups = pd.read_csv(cfg["data"]["patterns_path"], dtype={"StockCode": str})
    return latest.merge(groups[["StockCode", "group"]], on="StockCode", how="left")


def normalise_code(series: pd.Series) -> pd.Series:
    """Same rule as the cleaning pipeline: strip padding, force uppercase."""
    return series.astype(str).str.strip().str.upper()


def build_template(skus) -> bytes:
    """Excel file listing every part, with a blank stock column to fill in."""
    tpl = pd.DataFrame({"StockCode": sorted(skus), "on_hand": 0})
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        tpl.to_excel(writer, index=False, sheet_name="stock")
    return buffer.getvalue()


def read_upload(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


bundle = load_model()
data = score_latest()
# Read from the bundle, not config: these are the settings the model was trained for.
lead_time = bundle["lead_time_weeks"]
service = bundle["service_level"]
known_skus = set(data["StockCode"])

st.title("Parts Reorder Optimiser")
st.caption(f"Lead time {lead_time} weeks · target service level {service:.0%} "
           f"· forecast week of {data['week'].max().date()}")

# ---------------- sidebar ----------------
st.sidebar.header("Stock on hand")
st.sidebar.download_button(
    "Download blank template",
    build_template(known_skus),
    "stock_template.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
uploaded = st.sidebar.file_uploader("Upload stock file", type=["xlsx", "xls", "csv"])

st.sidebar.header("Order settings")
cover = st.sidebar.slider("Weeks of cover to order", 2, 12, 4)
show = st.sidebar.selectbox("Show", ["Order now", "All products"])

st.sidebar.header("Model")
st.sidebar.caption(f"Trained {bundle['trained_at'].replace('T', ' ')} · "
                   f"data to {bundle['train_end']} · "
                   f"test coverage {bundle['metrics']['coverage']:.1%}")

# ---------------- attach stock levels ----------------
if uploaded is None:
    rng = np.random.default_rng(42)
    data["on_hand"] = (data["reorder_point"] * rng.uniform(0.4, 1.6, len(data))).round()
    st.info("Demo mode: stock levels are simulated. "
            "Upload a stock file to use real figures.")
else:
    try:
        raw = read_upload(uploaded)
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            st.error(f"File is missing required column(s): {', '.join(missing)}. "
                     "Download the template to see the expected format.")
            st.stop()

        raw = raw[REQUIRED_COLUMNS].copy()
        raw["StockCode"] = normalise_code(raw["StockCode"])
        raw["on_hand"] = pd.to_numeric(raw["on_hand"], errors="coerce")

        bad_numbers = raw["on_hand"].isna().sum()
        raw = raw.dropna(subset=["on_hand"])
        raw = raw.drop_duplicates(subset="StockCode", keep="last")

        unknown = sorted(set(raw["StockCode"]) - known_skus)
        data = data.merge(raw, on="StockCode", how="left")
        not_supplied = data["on_hand"].isna().sum()
        data["on_hand"] = data["on_hand"].fillna(0)

        st.success(f"Loaded stock for {len(raw) - len(unknown):,} of "
                   f"{len(known_skus):,} parts.")
        if bad_numbers:
            st.warning(f"{bad_numbers} row(s) had a non-numeric stock value and were ignored.")
        if not_supplied:
            st.warning(f"{not_supplied} part(s) had no stock row and were treated as zero.")
        if unknown:
            st.warning(f"{len(unknown)} code(s) in the file are not in scope and were "
                       f"ignored: {', '.join(unknown[:8])}"
                       f"{' ...' if len(unknown) > 8 else ''}")
    except Exception as err:
        st.error(f"Could not read the file: {err}")
        st.stop()

# ---------------- the reorder rule ----------------
d = data.copy()
d["order_qty"] = (d["reorder_point"] + cover * d["avg_4"] - d["on_hand"]).clip(lower=0).round()
d["shortfall"] = (d["reorder_point"] - d["on_hand"]).round()
d["status"] = np.where(d["on_hand"] <= d["reorder_point"], "ORDER NOW",
              np.where(d["on_hand"] <= d["reorder_point"] * 1.2, "Watch", "OK"))

urgent = d[d["status"] == "ORDER NOW"]

c1, c2, c3 = st.columns(3)
c1.metric("Products to order", f"{len(urgent):,}")
c2.metric("Units to order", f"{urgent['order_qty'].sum():,.0f}")
c3.metric("Watch list", f"{(d['status'] == 'Watch').sum():,}")

view = (urgent if show == "Order now" else d).sort_values("shortfall", ascending=False)

st.dataframe(
    view[["StockCode", "group", "on_hand", "pred_lgbm",
          "reorder_point", "order_qty", "status"]]
    .rename(columns={
        "StockCode": "Part",
        "group": "Type",
        "on_hand": "On hand",
        "pred_lgbm": f"Forecast ({lead_time}w)",
        "reorder_point": "Reorder point",
        "order_qty": "Order qty",
        "status": "Status",
    }).round(0),
    width="stretch", hide_index=True,
)

st.download_button("Download order list",
                   urgent.to_csv(index=False).encode(),
                   "reorder_list.csv", "text/csv")