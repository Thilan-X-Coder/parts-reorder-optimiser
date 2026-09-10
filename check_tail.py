"""Investigate the unusually large final week."""

import pandas as pd
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

raw = pd.read_csv(config["data"]["raw_path"])
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])

# Same cleaning as the pipeline
raw["StockCode"] = raw["StockCode"].astype(str).str.strip().str.upper()
clean = raw.drop_duplicates()
clean = clean[
    clean["StockCode"].str.fullmatch(r"^\d{5}[A-Z]*$")
    & (clean["Quantity"] > 0)
    & (clean["Price"] > 0)
]

tail = clean[clean["InvoiceDate"] >= "2011-11-01"].copy()

print("=== Daily demand, Nov 2011 onward ===")
print(tail.groupby(tail["InvoiceDate"].dt.date)["Quantity"].sum().to_string())

print("\n=== Biggest invoices in the final week ===")
last = clean[clean["InvoiceDate"] >= "2011-12-05"]
by_invoice = (last.groupby("Invoice")["Quantity"].sum()
                  .sort_values(ascending=False).head(10))
print(by_invoice.to_string())
print("\nFinal week total quantity:", last["Quantity"].sum())
print("Share from top 3 invoices :",
      round(100 * by_invoice.head(3).sum() / last["Quantity"].sum(), 1), "%")

print("\n=== Largest single rows in the final week ===")
print(last.nlargest(10, "Quantity")[
    ["Invoice", "StockCode", "Description", "Quantity", "Price", "InvoiceDate"]
].to_string())