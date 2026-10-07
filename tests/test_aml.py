import numpy as np
import pandas as pd

from aml.evaluate import evaluate_alerts, fit_score
from aml.features import GROUP_A, GROUP_B, GROUP_FLOW, build_features
from aml.rules import RULES, apply_rules
from aml.sim import CASH, DAYS, Network, simulate


def tiny(n, txns):
    acc = pd.DataFrame({"account_type": ["retail"] * n, "suspicious": [False] * n, "typology": [""] * n, "lookalike": [""] * n})
    return Network(acc, pd.DataFrame(txns, columns=["src", "dst", "amount", "day", "cash"]))


def test_simulator_is_deterministic_and_has_all_ingredients():
    a, b = simulate(6000, seed=3, n_structuring=5, n_layering=4, n_cycle=4, n_scatter=3), simulate(6000, seed=3, n_structuring=5, n_layering=4, n_cycle=4, n_scatter=3)
    assert a.txns.equals(b.txns) and a.accounts.equals(b.accounts)
    acc = a.accounts
    assert set(acc.loc[acc["suspicious"], "typology"]) == {"structuring", "layering", "cycle", "gather_scatter"}
    assert {"payroll", "processor", "merchant", "bill_split", "cash_business"} <= set(acc["lookalike"])
    # Payroll, processors, merchants and bill-splitters are never launderers; a cash-intensive shop can also be a mule.
    assert not (acc["suspicious"] & acc["lookalike"].isin(["payroll", "processor", "merchant", "bill_split"])).any()
    t = a.txns
    assert (t.loc[t["cash"] == 1, "src"] == CASH).all() and t["day"].between(0, DAYS - 1).all() and (t["amount"] > 0).all()


def test_stealth_makes_laundered_amounts_ordinary():
    def median_mule_transfer(stealth):
        net = simulate(6000, seed=4, stealth=stealth, n_structuring=0, n_layering=10, n_cycle=10, n_scatter=0)
        mules = set(net.accounts.index[net.accounts["suspicious"]])
        t = net.txns
        both = t[t["src"].isin(mules) & t["dst"].isin(mules) & (t["cash"] == 0) & (t["amount"] > 700)]
        return both["amount"].median()

    assert median_mule_transfer(0.0) > 5 * median_mule_transfer(1.0)


def test_features_detect_a_cycle_a_chain_and_structuring():
    txns = [(0, 1, 9000, 5, 0), (1, 2, 8800, 7, 0), (2, 0, 8600, 9, 0),  # a 3-cycle of large transfers
            (10, 11, 20000, 5, 0), (11, 12, 19000, 6, 0), (12, 13, 18000, 7, 0),  # a chain, hop within a day
            (CASH, 20, 9500, 1, 1), (CASH, 20, 9400, 2, 1), (CASH, 20, 9600, 3, 1), (20, 21, 28000, 4, 0)]  # structuring
    f = build_features(tiny(25, txns))
    assert f.loc[0, "tri_count"] == 1 and f.loc[[0, 1, 2], "big_scc_size"].eq(3).all() and f.loc[5, "big_scc_size"] == 0
    assert f.loc[11, "pass_through"] > 0.9 and f.loc[11, "flow_match_cnt"] == 1 and f.loc[10, "flow_match_cnt"] == 0
    assert f.loc[11, "n_hp_nbrs"] >= 1 and f.loc[10, "n_hp_nbrs"] >= 1  # neighbours of a pass-through account
    assert f.loc[20, "cash_near_threshold"] == 3 and abs(f.loc[20, "pass_through"] - 19_000 / 28_500) < 1e-3  # only deposits made 2+ days before the exit count
    assert not f.isna().any().any() and set(GROUP_A + GROUP_FLOW + GROUP_B) <= set(f.columns)


def test_rules_fire_where_expected():
    txns = [(CASH, 20, 9500, 1, 1), (CASH, 20, 9400, 2, 1), (CASH, 20, 9600, 3, 1), (30, 31, 50000, 4, 0)]
    hits = apply_rules(build_features(tiny(35, txns)))
    assert hits.loc[20, list(RULES)[0]] and hits.loc[30, list(RULES)[4]] and not hits.loc[5].any()


def test_alert_evaluation_arithmetic():
    acc = pd.DataFrame({"account_type": ["retail"] * 6, "suspicious": [True, True, True, False, False, False],
                        "typology": ["layering", "layering", "cycle", "", "", ""], "lookalike": ["", "", "", "payroll", "", ""]})  # fmt: skip
    r = evaluate_alerts(np.array([0, 2, 3, 4]), Network(acc, pd.DataFrame()))
    assert abs(r["recall"] - 2 / 3) < 1e-12 and r["precision"] == 0.5
    assert r["recall_by_typology"]["layering"] == 0.5 and r["recall_by_typology"]["cycle"] == 1.0
    assert r["false_positives"] == {"payroll": 1, "ordinary": 1}


def test_models_beat_rules_on_an_unseen_bank_and_network_features_help_cycles():
    kw = dict(n_structuring=10, n_layering=8, n_cycle=10, n_scatter=4, stealth=1.0)
    tr, te = simulate(12_000, seed=1, **kw), simulate(12_000, seed=2, **kw)
    Ftr, Fte = build_features(tr), build_features(te)
    ytr, yte = tr.accounts["suspicious"].to_numpy(), te.accounts["suspicious"].to_numpy()
    rules = apply_rules(Fte).any(axis=1).to_numpy()
    top = lambda cols: np.argsort(-fit_score(Ftr, ytr, Fte, cols))[: int(yte.sum() * 1.5)]
    cyc = (te.accounts["typology"] == "cycle").to_numpy()
    a, ab = top(GROUP_A), top(GROUP_A + GROUP_B)
    assert yte[a].mean() > 3 * yte[rules].mean()  # precision of the model's queue vs the rules'
    assert np.isin(np.flatnonzero(cyc), ab).mean() > np.isin(np.flatnonzero(cyc), a).mean()
