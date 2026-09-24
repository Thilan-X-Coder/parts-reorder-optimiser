import yaml
import pandas as pd
import src.policy as policy

with open("config.yaml") as f:
    config = yaml.safe_load(f)

inv = config["inventory"]
lead_time = inv["lead_time_weeks"]

test = pd.read_csv("data/processed/predictions.csv", parse_dates=["week"])

# Current practice: 4-week average x lead time, plus a flat 25% buffer
test["rop_manual"] = test["avg_4"] * lead_time * 1.25

results = {
    "Manual rule": policy.simulate(test, "rop_manual", lead_time),
    "ML quantile": policy.simulate(test, "reorder_point", lead_time),
}
print("Calendar weeks in test period:", results["Manual rule"]["total_weeks"])
# ---------- service and stock ----------
print(f"{'Policy':<14}{'Fill rate':>12}{'Avg stock':>14}{'Stockout wks':>15}")
print("-" * 55)
for name, r in results.items():
    print(f"{name:<14}{r['fill_rate']:>11.1%}"
          f"{r['avg_stock_units']:>14,.0f}{r['stockout_weeks']:>15,}")

m, ml = results["Manual rule"], results["ML quantile"]
print(f"\nFill rate change : {ml['fill_rate'] - m['fill_rate']:+.1%}")
print(f"Stock change     : {(ml['avg_stock_units'] / m['avg_stock_units'] - 1):+.1%}")

# ---------- money ----------
print(f"\n{'Policy':<14}{'Holding':>12}{'Shortage':>12}{'Total':>12}")
print("-" * 50)
for name, r in results.items():
    c = policy.cost(r, inv["unit_cost"], inv["holding_cost_rate"],
                    inv["stockout_cost_per_unit"])
    print(f"{name:<14}{c['holding']:>12,.0f}{c['shortage']:>12,.0f}{c['total']:>12,.0f}")

# ---------- does the answer survive different assumptions? ----------
print(f"\n{'Stockout cost':<16}{'Manual':>14}{'ML':>14}{'Winner':>10}")
print("-" * 56)
for sc in [1.0, 2.5, 5.0, 7.5, 15.0]:
    cm = policy.cost(results["Manual rule"], inv["unit_cost"],
                     inv["holding_cost_rate"], sc)["total"]
    cl = policy.cost(results["ML quantile"], inv["unit_cost"],
                     inv["holding_cost_rate"], sc)["total"]
    print(f"{sc:<16.2f}{cm:>14,.0f}{cl:>14,.0f}{'ML' if cl < cm else 'Manual':>10}")


print(f"\n{'Manual buffer':<16}{'Fill rate':>12}{'Avg stock':>14}{'Total cost':>14}")
print("-" * 58)
for mult in [1.25, 1.5, 1.75, 2.0, 2.5]:
    test["rop_tuned"] = test["avg_4"] * lead_time * mult
    r = policy.simulate(test, "rop_tuned", lead_time)
    c = policy.cost(r, inv["unit_cost"], inv["holding_cost_rate"],
                    inv["stockout_cost_per_unit"])
    print(f"{mult:<16.2f}{r['fill_rate']:>11.1%}"
          f"{r['avg_stock_units']:>14,.0f}{c['total']:>14,.0f}")

print(f"{'ML quantile':<16}{ml['fill_rate']:>11.1%}"
      f"{ml['avg_stock_units']:>14,.0f}"
      f"{policy.cost(ml, inv['unit_cost'], inv['holding_cost_rate'], inv['stockout_cost_per_unit'])['total']:>14,.0f}")