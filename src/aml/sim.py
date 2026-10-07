"""A synthetic bank: accounts, legitimate payments, benign look-alikes of money laundering, and four laundering typologies.

Typologies (positives, labelled per account):
  structuring   many cash deposits just under the 10,000 reporting threshold, then one transfer out to a collector
  layering      funds hop through a chain of mule accounts, each forwarding ~95% within 0-2 days
  cycle         money travels A -> B -> C (-> D) -> A with slowly shrinking amounts
  gather_scatter many small payers -> one hub -> many receivers (only the hub is labelled)
Benign look-alikes (they trigger naive rules; only a cash-intensive shop can also be a mule): payroll runs, payment processors that forward
98% of what they receive, merchants with many payers, cash-intensive shops, friends splitting a bill in a triangle.
Mules also make a few ordinary payments, as real ones do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS = 60
CASH_THRESHOLD = 10_000.0
CASH = -1  # virtual source of cash deposits


@dataclass
class Network:
    accounts: pd.DataFrame  # account_type, suspicious, typology, lookalike
    txns: pd.DataFrame  # src, dst, amount, day, cash


def simulate(n_accounts: int = 30_000, seed: int = 0, stealth: float = 0.0, n_structuring=25, n_layering=15, n_cycle=15, n_scatter=10) -> Network:
    """stealth in [0, 1]: how well launderers hide. At 0 they move large, obvious sums (median ~35,000); at 1 the laundered
    amounts are comparable to what an ordinary small-business account moves (median ~2,500), hops are slower (up to 6 days)
    and structured deposits are no longer all just below the reporting threshold. (A first version added extra ordinary
    activity to mules instead; that made them MORE conspicuous and is not what hiding means.)"""
    rng = np.random.default_rng(seed)
    n = n_accounts
    kind = rng.choice(["retail", "small_biz", "cash_biz"], n, p=[0.85, 0.12, 0.03])
    lam = np.select([kind == "retail", kind == "small_biz"], [6, 25], 20)
    mu = np.select([kind == "retail", kind == "small_biz"], [np.log(150), np.log(800)], np.log(1500))
    rows: list[tuple] = []  # (src, dst, amount, day, cash)

    # Legitimate transfers: mostly repeat partners, the rest towards popular accounts (merchants).
    pop = rng.pareto(1.5, n) + 1
    pop /= pop.sum()
    partners = [rng.choice(n, size=rng.integers(3, 7), p=pop) for _ in range(n)]
    n_out = rng.poisson(lam)
    for i in range(n):
        k = n_out[i]
        if k == 0:
            continue
        dst = np.where(rng.random(k) < 0.7, rng.choice(partners[i], k), rng.choice(n, k, p=pop))
        amt = rng.lognormal(mu[i], 1.0, k).round(2)
        day = rng.integers(0, DAYS, k)
        rows.extend((i, int(d), float(a), int(t), 0) for d, a, t in zip(dst, amt, day) if d != i)
    # Legitimate cash deposits.
    k_cash = rng.poisson(np.select([kind == "retail", kind == "small_biz"], [0.8, 4], 25))
    cash_mu = np.select([kind == "retail", kind == "small_biz"], [np.log(600), np.log(2000)], np.log(6000))
    for i in np.flatnonzero(k_cash):
        amt = np.minimum(rng.lognormal(cash_mu[i], 0.7, k_cash[i]), 60_000).round(2)
        rows.extend((CASH, int(i), float(a), int(rng.integers(0, DAYS)), 1) for a in amt)

    suspicious = np.zeros(n, bool)
    typology = np.array([""] * n, dtype=object)
    lookalike = np.array([""] * n, dtype=object)
    free = list(rng.permutation(n))

    def take(k):
        return [free.pop() for _ in range(k)]

    def tag_pos(ids, name):
        suspicious[ids] = True
        typology[ids] = name

    median_amount = float(np.exp((1 - stealth) * np.log(35_000) + stealth * np.log(2_500)))

    def laundered_amount() -> float:
        return float(np.clip(rng.lognormal(np.log(median_amount), 0.4), 800, 90_000))

    def camouflage(ids):  # a few ordinary payments, as every real account has
        for a in ids:
            for _ in range(rng.poisson(3)):
                other = int(rng.choice(n, p=pop))
                src, dst = (a, other) if rng.random() < 0.5 else (other, a)
                if src != dst:
                    rows.append((int(src), int(dst), float(rng.lognormal(np.log(150), 1.0)), int(rng.integers(0, DAYS)), 0))

    # ---- benign look-alikes (counts scale with the size of the bank; the defaults are for 30,000 accounts)
    scale = n / 30_000
    k = lambda base: max(1, round(base * scale))
    for _ in range(k(40)):  # payroll: one business pays 15-60 people on two paydays
        biz, staff = take(1)[0], take(int(rng.integers(15, 60)))
        lookalike[biz] = "payroll"
        base = rng.lognormal(np.log(2500), 0.3, len(staff))
        for day in (27, 57):
            rows.extend((biz, int(s), float(b * rng.uniform(0.97, 1.03)), day, 0) for s, b in zip(staff, base))
    for _ in range(k(40)):  # payment processors: many payers in, forwards 98% of the day's inflow to the same settlement account
        proc, settle, payers = take(1)[0], take(1)[0], take(int(rng.integers(20, 60)))
        lookalike[proc] = "processor"
        for day in range(0, DAYS, 2):
            batch = rng.lognormal(np.log(400), 0.8, int(rng.integers(3, 8)))
            for a, p in zip(batch, rng.choice(payers, len(batch))):
                rows.append((int(p), proc, float(a), day, 0))
            rows.append((proc, settle, float(batch.sum() * 0.98), day + int(rng.integers(0, 2)), 0))
    for _ in range(k(60)):  # merchants with many payers
        m, payers = take(1)[0], take(int(rng.integers(20, 80)))
        lookalike[m] = "merchant"
        rows.extend((int(p), m, float(rng.lognormal(np.log(60), 0.6)), int(rng.integers(0, DAYS)), 0) for p in payers)
    for _ in range(k(150)):  # friends splitting a bill: a small triangle of similar amounts
        a, b, c = take(3)
        amt, day = float(rng.lognormal(np.log(80), 0.4)), int(rng.integers(0, DAYS - 4))
        for s, d, dd in ((a, b, 0), (b, c, 1), (c, a, 2)):
            rows.append((s, d, amt * rng.uniform(0.95, 1.05), day + dd, 0))
        lookalike[[a, b, c]] = "bill_split"
    cb = np.flatnonzero((kind == "cash_biz") & (lookalike == ""))
    lookalike[cb] = "cash_business"

    # ---- laundering typologies
    for _ in range(n_structuring):
        smurf, collector = take(2)
        tag_pos([smurf, collector], "structuring")
        days = np.sort(rng.integers(0, DAYS - 6, int(rng.integers(5, 12))))
        amts = rng.uniform(8_000 - 3_000 * stealth, 9_900, len(days)).round(2)
        rows.extend((CASH, smurf, float(a), int(d), 1) for a, d in zip(amts, days))
        rows.append((smurf, collector, float(amts.sum() * 0.97), int(days[-1] + rng.integers(1, 4)), 0))
        camouflage([smurf, collector])
    for _ in range(n_layering):
        chain = take(int(rng.integers(4, 8)))
        tag_pos(chain, "layering")
        amt, day = laundered_amount(), int(rng.integers(0, DAYS - 14 - round(26 * stealth)))
        for s, d in zip(chain[:-1], chain[1:]):
            amt *= rng.uniform(0.93, 0.97)
            day += int(rng.integers(0, 3 + round(4 * stealth)))
            rows.append((s, d, round(amt, 2), day, 0))
        camouflage(chain)
    for _ in range(n_cycle):
        ring = take(int(rng.integers(3, 6)))
        tag_pos(ring, "cycle")
        amt, day = laundered_amount(), int(rng.integers(0, DAYS - 20 - round(20 * stealth)))
        for s, d in zip(ring, ring[1:] + ring[:1]):
            amt *= rng.uniform(0.97, 0.995)
            day += int(rng.integers(1, 5 + round(4 * stealth)))
            rows.append((s, d, round(amt, 2), day, 0))
        camouflage(ring)
    for _ in range(n_scatter):
        hub, sources, dests = take(1)[0], take(int(rng.integers(10, 25))), take(int(rng.integers(8, 15)))
        tag_pos([hub], "gather_scatter")
        day = int(rng.integers(0, DAYS - 14))
        got = 0.0
        for s in sources:
            a = float(np.clip(rng.lognormal(np.log(median_amount / 7), 0.4), 150, 10_000))
            got += a
            rows.append((s, hub, round(a, 2), day + int(rng.integers(0, 5)), 0))
        share = rng.dirichlet(np.ones(len(dests))) * got * 0.96
        rows.extend((hub, d, round(float(v), 2), day + int(rng.integers(5, 12)), 0) for d, v in zip(dests, share))
        camouflage([hub])

    t = pd.DataFrame(rows, columns=["src", "dst", "amount", "day", "cash"])
    t["day"] = t["day"].clip(0, DAYS - 1)
    accounts = pd.DataFrame({"account_type": kind, "suspicious": suspicious, "typology": typology, "lookalike": lookalike})
    return Network(accounts, t)
