import yaml
import pandas as pd
import src.classify as classify

with open("config.yaml") as f:
    config = yaml.safe_load(f)

weekly = pd.read_parquet(config["data"]["weekly_path"])
result = classify.classify(weekly)
result.to_csv(config["data"]["patterns_path"], index=False)

print(result["group"].value_counts())
print()
print(result.groupby("group")[["avg_demand", "zero_weeks"]].mean().round(2))