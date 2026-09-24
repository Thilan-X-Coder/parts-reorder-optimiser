import yaml
import pandas as pd
import src.split as split
import src.baseline as baseline
import src.model as model

with open("config.yaml") as f:
    config = yaml.safe_load(f)

lead_time = config["inventory"]["lead_time_weeks"]
service_level = config["inventory"]["service_level"]

df = pd.read_csv(config["data"]["features_path"], parse_dates=["week"])
groups = pd.read_csv(config["data"]["patterns_path"])
df = df.merge(groups[["StockCode", "group"]], on="StockCode", how="left")

train, test = split.time_split(df, config["split"]["train_end"], lead_time)
print(f"Train {len(train):,} rows | Test {len(test):,} rows")

# ---------- train every point-forecast model ----------
test["pred_base"] = baseline.predict(test, lead_time)

models = {}
for kind in ["lgbm", "rf"]:
    models[kind] = model.train(train, kind)
    test[f"pred_{kind}"] = model.predict(models[kind], test)

columns = {"Baseline": "pred_base", "LightGBM": "pred_lgbm", "RandomForest": "pred_rf"}

# ---------- overall metrics ----------
print(f"\n{'Model':<14}{'MAE':>10}{'RMSE':>12}{'Bias':>10}{'vs base':>10}")
print("-" * 56)
base_mae = baseline.score(test["target"], test["pred_base"])["MAE"]

for name, col in columns.items():
    s = baseline.score(test["target"], test[col])
    change = (s["MAE"] - base_mae) / base_mae
    print(f"{name:<14}{s['MAE']:>10,.1f}{s['RMSE']:>12,.1f}"
          f"{s['Bias']:>10,.1f}{change:>9.1%}")

print(f"\nMean actual demand over {lead_time} weeks: {test['target'].mean():,.1f}")

# ---------- MAE by product group ----------
print(f"\n{'Group':<10}{'n':>8}" + "".join(f"{n:>14}" for n in columns))
print("-" * 60)
for group in ["fast", "slow"]:
    part = test[test["group"] == group]
    row = "".join(f"{baseline.score(part['target'], part[c])['MAE']:>14,.1f}"
                  for c in columns.values())
    print(f"{group:<10}{len(part):>8,}{row}")

# ---------- what the winning model learned ----------
print("\nLightGBM top features:")
print(model.importance(models["lgbm"]).head(6).to_string(index=False))

# ---------- reorder point: quantile model only ----------
qm = model.train(train, "quantile", service_level)
test["reorder_point"] = model.predict(qm, test)
covered = (test["target"] <= test["reorder_point"]).mean()

print(f"\nQuantile model (target {service_level:.0%} service level)")
print(f"  Actual coverage : {covered:.1%}")
print(f"  Mean forecast   : {test['pred_lgbm'].mean():,.0f}")
print(f"  Mean reorder pt : {test['reorder_point'].mean():,.0f}")

test.to_csv("data/processed/predictions.csv", index=False)
print("\nSaved: data/processed/predictions.csv")