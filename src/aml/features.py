"""Features in two groups, so the value of the network can be measured separately.

Group A, account behaviour (what a classic transaction-monitoring system computes per account):
  counts and amounts in and out, cash deposits and how many sit just below the reporting threshold,
  distinct counterparties, concentration, and how much of the incoming money leaves again within two days.
Group "flow" is one smarter account-level feature: how many large incoming transfers were followed, within 15 days,
by an outgoing one of 85-100% of the amount (large = at least 1,500) (a conservation-of-money signature that does not depend on timing).
Group B, network context (what only looks at the *neighbours*):
  triangles, membership of a strongly connected group of large transfers (money going round in circles),
  PageRank, and how many of an account's counterparties are themselves fast pass-through accounts.
All features use the transactions of the window only (no future information).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .sim import CASH_THRESHOLD, DAYS, Network

GROUP_A = ["out_cnt", "in_cnt", "out_sum", "in_sum", "max_out", "max_in", "cash_cnt", "cash_sum", "cash_near_threshold",
           "n_out_cp", "n_in_cp", "hhi_out", "pass_through", "throughput", "out_in_ratio", "acct_small_biz", "acct_cash_biz"]  # fmt: skip
GROUP_FLOW = ["flow_match_cnt"]
GROUP_B = ["tri_count", "big_scc_size", "pagerank", "nbr_mean_pass", "nbr_max_pass", "n_hp_nbrs", "n_hp_in_nbrs"]


def build_features(net: Network, big_edge: float = 1_500.0) -> pd.DataFrame:
    n = len(net.accounts)
    t = net.txns
    tr, cash = t[t["cash"] == 0], t[t["cash"] == 1]
    src, dst, amt, day = tr["src"].to_numpy(), tr["dst"].to_numpy(), tr["amount"].to_numpy(), tr["day"].to_numpy()

    f = pd.DataFrame(index=np.arange(n))
    f["out_cnt"], f["in_cnt"] = np.bincount(src, minlength=n), np.bincount(dst, minlength=n)
    f["out_sum"], f["in_sum"] = np.bincount(src, amt, n), np.bincount(dst, amt, n)
    f["max_out"] = pd.Series(amt).groupby(src).max().reindex(f.index).fillna(0).to_numpy()
    f["max_in"] = pd.Series(amt).groupby(dst).max().reindex(f.index).fillna(0).to_numpy()
    c_dst, c_amt = cash["dst"].to_numpy(), cash["amount"].to_numpy()
    f["cash_cnt"], f["cash_sum"] = np.bincount(c_dst, minlength=n), np.bincount(c_dst, c_amt, n)
    near = (c_amt >= 0.8 * CASH_THRESHOLD) & (c_amt < CASH_THRESHOLD)
    f["cash_near_threshold"] = np.bincount(c_dst[near], minlength=n)

    pairs = pd.DataFrame({"s": src, "d": dst, "a": amt}).groupby(["s", "d"], as_index=False)["a"].sum()
    ps, pd_, pa = pairs["s"].to_numpy(), pairs["d"].to_numpy(), pairs["a"].to_numpy()
    f["n_out_cp"], f["n_in_cp"] = np.bincount(ps, minlength=n), np.bincount(pd_, minlength=n)
    f["hhi_out"] = np.bincount(ps, (pa / np.maximum(f["out_sum"].to_numpy()[ps], 1e-9)) ** 2, n)

    # Money in (transfers and cash) against money out within the next two days.
    inflow, outflow = np.zeros((n, DAYS), np.float32), np.zeros((n, DAYS), np.float32)
    np.add.at(inflow, (dst, day), amt)
    np.add.at(inflow, (c_dst, cash["day"].to_numpy()), c_amt)
    np.add.at(outflow, (src, day), amt)
    ahead = outflow.copy()
    ahead[:, :-1] += outflow[:, 1:]
    ahead[:, :-2] += outflow[:, 2:]
    moved = np.minimum(inflow, ahead).sum(axis=1)
    total_in = inflow.sum(axis=1)
    f["pass_through"] = np.where(total_in > 0, moved / np.maximum(total_in, 1e-9), 0.0)
    f["throughput"] = moved
    f["out_in_ratio"] = f["out_sum"] / (f["in_sum"] + f["cash_sum"] + 1.0)
    kind = net.accounts["account_type"].to_numpy()
    f["acct_small_biz"], f["acct_cash_biz"] = (kind == "small_biz").astype(int), (kind == "cash_biz").astype(int)

    big_tx = tr[tr["amount"] >= big_edge][["src", "dst", "amount", "day"]]
    m = big_tx.rename(columns={"src": "s_in", "amount": "a_in", "day": "d_in"}).merge(
        big_tx.rename(columns={"dst": "d_out", "amount": "a_out", "day": "d_out_day"}), left_on="dst", right_on="src")  # fmt: skip
    ratio = m["a_out"] / m["a_in"]
    ok = (ratio >= 0.85) & (ratio <= 1.0) & (m["d_out_day"] >= m["d_in"]) & (m["d_out_day"] - m["d_in"] <= 15) & (m["s_in"] != m["d_out"])
    f["flow_match_cnt"] = np.bincount(m.loc[ok, "dst"].to_numpy(), minlength=n)

    # ---- group B
    A = sparse.csr_matrix((np.ones(len(ps)), (ps, pd_)), shape=(n, n))
    f["tri_count"] = np.asarray(((A @ A).multiply(A.T)).sum(axis=1)).ravel()
    keep = pa >= big_edge
    big = sparse.csr_matrix((np.ones(keep.sum()), (ps[keep], pd_[keep])), shape=(n, n))
    _, labels = connected_components(big, directed=True, connection="strong")
    sizes = np.bincount(labels)
    f["big_scc_size"] = np.where(sizes[labels] >= 2, sizes[labels], 0)
    W = sparse.csr_matrix((pa, (ps, pd_)), shape=(n, n))
    out_w = np.asarray(W.sum(axis=1)).ravel()
    P = sparse.diags(1 / np.maximum(out_w, 1e-9)) @ W  # row-stochastic
    rank = np.full(n, 1 / n)
    for _ in range(25):
        rank = 0.15 / n + 0.85 * (P.T @ rank + rank[out_w == 0].sum() / n)
    f["pagerank"] = rank * n

    sym = ((A + A.T) > 0).astype(float).tocsr()
    deg = np.maximum(np.asarray(sym.sum(axis=1)).ravel(), 1)
    pas = f["pass_through"].to_numpy()
    hp = ((pas > 0.8) & (f["throughput"].to_numpy() > 5_000)).astype(float)
    f["nbr_mean_pass"] = (sym @ pas) / deg
    f["nbr_max_pass"] = np.asarray(sym.multiply(pas[None, :]).max(axis=1).todense()).ravel()
    f["n_hp_nbrs"] = sym @ hp
    f["n_hp_in_nbrs"] = A.T @ hp
    return f
