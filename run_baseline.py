import yaml
import pandas as pd
import src.slpit as split
import src.baseline as baseline

with open("config.yaml") as f:
    config = yaml.safe_load(f)

lead_time = config["inventory"]["lead_time_weeks"]

df = pd.read_csv(config["data"]["features_path"], parse_dates=["week"])
train, test = split.time_split(df, config["split"]["train_end"], lead_time)

print(f"Train: {len(train):,} rows  ({train.week.min().date()} to {train.week.max().date()})")
print(f"Test : {len(test):,} rows  ({test.week.min().date()} to {test.week.max().date()})")
print()

pred = baseline.predict(test, lead_time)
results = baseline.score(test["target"], pred)

for k, v in results.items():
    print(f"{k:12} {v:>10,.1f}")