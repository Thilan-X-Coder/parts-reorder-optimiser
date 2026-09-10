import yaml
import pandas as pd
import src.features as features

with open("config.yaml") as f:
    config = yaml.safe_load(f)

lead_time = config["inventory"]["lead_time_weeks"]

weekly = pd.read_parquet(config["data"]["weekly_path"])

df = features.build_features(weekly)
df = features.add_target(df, lead_time)

df.to_csv(config["data"]["features_path"], index=False)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Weeks:", df["week"].min().date(), "to", df["week"].max().date())
