"""Build the weekly demand table from the raw transaction file."""

import yaml
import src.data as data

with open("config.yaml") as f:
    config = yaml.safe_load(f)

data.build(config)