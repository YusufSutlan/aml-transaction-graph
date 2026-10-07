"""The classic rule-based transaction monitoring that most banks run today: simple thresholds, one rule per typology."""

from __future__ import annotations

import pandas as pd

RULES = {
    "structuring: >=3 cash deposits of 8,000-9,999": lambda f: f["cash_near_threshold"] >= 3,
    "large cash: >= 50,000 deposited": lambda f: f["cash_sum"] >= 50_000,
    "rapid pass-through: >=85% forwarded, >= 20,000": lambda f: (f["pass_through"] >= 0.85) & (f["throughput"] >= 20_000),
    "fan-in / fan-out: >=15 payers, >=5 receivers": lambda f: (f["n_in_cp"] >= 15) & (f["n_out_cp"] >= 5) & (f["in_sum"] >= 20_000),
    "single transfer >= 30,000": lambda f: f["max_out"] >= 30_000,
}


def apply_rules(f: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: rule(f) for name, rule in RULES.items()})
