import yaml
import pandas as pd
import src.split as split
import src.model as model
import src.artifacts as artifacts

BUNDLE_PATH = "models/bundle.pkl"

with open("config.yaml") as f:
    config = yaml.safe_load(f)

lead_time = config["inventory"]["lead_time_weeks"]
service_level = config["inventory"]["service_level"]

df = pd.read_csv(config["data"]["features_path"], parse_dates=["week"])
train, test = split.time_split(df, config["split"]["train_end"], lead_time)
print(f"Train {len(train):,} rows | Test {len(test):,} rows")

point_model = model.train(train, "lgbm")
quantile_model = model.train(train, "quantile", service_level)

pred = model.predict(point_model, test)
reorder_point = model.predict(quantile_model, test)
error = pred - test["target"]

metrics = {
    "mae": float(error.abs().mean()),
    "bias": float(error.mean()),
    "coverage": float((test["target"] <= reorder_point).mean()),
    "test_rows": len(test),
}

artifacts.save_bundle(BUNDLE_PATH, point_model, quantile_model,
                      model.FEATURES, config, metrics)

print(f"\nMAE      : {metrics['mae']:,.1f}")
print(f"Bias     : {metrics['bias']:,.1f}")
print(f"Coverage : {metrics['coverage']:.1%} (target {service_level:.0%})")
print(f"\nSaved: {BUNDLE_PATH}")
