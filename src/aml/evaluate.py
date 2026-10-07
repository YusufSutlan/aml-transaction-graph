"""Rules vs. account-level models vs. models with network features, on a bank the models have never seen.

Because the answer depends on how well launderers hide, everything is run at three stealth levels and the whole
curve is reported. Each repetition simulates a training bank and a different test bank (3 repetitions).

Usage:
    python -m aml.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from .features import GROUP_A, GROUP_B, GROUP_FLOW, build_features
from .rules import apply_rules
from .sim import simulate

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
BUDGET = 300  # 1% of 30,000 accounts: what an investigation team can review
TYPOLOGIES = ["structuring", "layering", "cycle", "gather_scatter"]
N_REPS = 3
STEALTH = (0.0, 0.5, 1.0)
MODELS = {
    "account (A)": GROUP_A,
    "account + flow match": GROUP_A + GROUP_FLOW,
    "account + network (A+B)": GROUP_A + GROUP_B,
    "account + flow + network": GROUP_A + GROUP_FLOW + GROUP_B,
    "network only (B)": GROUP_B,
}


def make_rep(i: int, stealth: float = 0.0):
    tr, te = simulate(seed=10 + i, stealth=stealth), simulate(seed=20 + i, stealth=stealth)
    return tr, te, build_features(tr), build_features(te)


def fit_score(Ftr, ytr, Fte, cols, seed=0):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    return m.fit(Ftr[cols], ytr).predict_proba(Fte[cols])[:, 1]


def evaluate_alerts(alert_idx: np.ndarray, net) -> dict:
    a = net.accounts
    y = a["suspicious"].to_numpy()
    hit = np.zeros(len(a), bool)
    hit[alert_idx] = True
    out = {"alerts": int(hit.sum()), "recall": float(hit[y].mean()), "precision": float(y[hit].mean()) if hit.any() else 0.0}
    out["recall_by_typology"] = {t: float(hit[m].mean()) if m.any() else float("nan") for t in TYPOLOGIES for m in [(a["typology"] == t).to_numpy()]}
    fp = a[hit & ~y]["lookalike"].replace("", "ordinary").value_counts()
    out["false_positives"] = {k: int(v) for k, v in fp.items()}
    return out


def run_rep(rep, coverage: float = 1.0, seed: int = 0) -> dict:
    tr, te, Ftr, Fte = rep
    ytr = tr.accounts["suspicious"].to_numpy().copy()
    if coverage < 1:  # only some launderers have been caught: the rest sit in the data as "negatives"
        pos = np.flatnonzero(ytr)
        ytr[np.random.default_rng(seed).choice(pos, int(round((1 - coverage) * len(pos))), replace=False)] = False
    yte = te.accounts["suspicious"].to_numpy()
    res = {"rules": evaluate_alerts(np.flatnonzero(apply_rules(Fte).any(axis=1).to_numpy()), te)}
    n_rules = res["rules"]["alerts"]
    scores = {name: fit_score(Ftr, ytr, Fte, cols, seed) for name, cols in MODELS.items()}
    for name, s in scores.items():
        order = np.argsort(-s)
        res[name] = {**evaluate_alerts(order[:BUDGET], te), "pr_auc": float(average_precision_score(yte, s)),
                     "recall_at_rules_volume": float(yte[order[:n_rules]].sum() / yte.sum())}  # fmt: skip
    res["_scores"] = scores
    return res


COLORS = {name: f"C{i}" for i, name in enumerate(MODELS)}


def ms(vals):
    return float(np.mean(vals)), float(np.std(vals))


def plot_stealth(by_stealth) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for name in ("account (A)", "account + flow match", "account + network (A+B)", "account + flow + network", "network only (B)"):
        m = [ms([r[name]["recall"] for r in by_stealth[s]]) for s in STEALTH]
        axes[0].errorbar(STEALTH, [v[0] for v in m], yerr=[v[1] for v in m], marker="o", capsize=3, label=name, color=COLORS[name])
    rules = [ms([r["rules"]["recall"] for r in by_stealth[s]])[0] for s in STEALTH]
    axes[0].plot(STEALTH, rules, "k--", marker="s", label=f"rules ({np.mean([r['rules']['alerts'] for r in by_stealth[0.0]]):.0f} alerts)")
    axes[0].set_xlabel("launderer stealth (0 = obvious, 1 = well hidden)")
    axes[0].set_ylabel(f"recall (models: top {BUDGET} alerts)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    for name in ("account (A)", "account + network (A+B)", "account + flow + network"):
        m = [ms([r[name]["pr_auc"] for r in by_stealth[s]]) for s in STEALTH]
        axes[1].errorbar(STEALTH, [v[0] for v in m], yerr=[v[1] for v in m], marker="o", capsize=3, label=name, color=COLORS[name])
    axes[1].set_xlabel("launderer stealth")
    axes[1].set_ylabel("PR-AUC")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "stealth_sweep.png", dpi=130)
    plt.close(fig)


def plot_typology(by_stealth, stealth=1.0) -> None:
    methods = ["rules", "account (A)", "account + flow match", "account + network (A+B)", "account + flow + network"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.16
    for i, m in enumerate(methods):
        vals = [ms([r[m]["recall_by_typology"][t] for r in by_stealth[stealth]]) for t in TYPOLOGIES]
        ax.bar(np.arange(4) + (i - 2) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], capsize=2, label=m, color=COLORS.get(m, "k"))
    ax.set_xticks(range(4), TYPOLOGIES)
    ax.set_ylabel("share of accounts caught")
    ax.set_title(f"Recall by typology at stealth {stealth:g} (models: top {BUDGET}; rules: all their alerts)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "typology_recall.png", dpi=130)
    plt.close(fig)


def plot_coverage(cov) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("account (A)", "account + flow + network"):
        xs = sorted(cov)
        m = [ms([r[name]["recall"] for r in cov[c]]) for c in xs]
        ax.errorbar([c * 100 for c in xs], [v[0] for v in m], yerr=[v[1] for v in m], marker="o", capsize=3, label=name, color=COLORS[name])
    ax.set_xlabel("share of true launderers labelled in the training data (%)")
    ax.set_ylabel(f"recall in the top {BUDGET} alerts (stealth 0.5)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "label_coverage.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    reps = {s: [make_rep(i, s) for i in range(N_REPS)] for s in STEALTH}
    by_stealth = {s: [run_rep(r) for r in reps[s]] for s in STEALTH}

    print(f"{'recall (top 300; rules: all alerts)':<40}" + "".join(f"{'stealth ' + str(s):>16}" for s in STEALTH))
    print(f"{'rules (5 rules, ~' + str(int(np.mean([r['rules']['alerts'] for r in by_stealth[0.0]]))) + ' alerts)':<40}" + "".join(f"{ms([r['rules']['recall'] for r in by_stealth[s]])[0]:>16.2f}" for s in STEALTH))
    for name in MODELS:
        print(f"{name:<40}" + "".join(f"{ms([r[name]['recall'] for r in by_stealth[s]])[0]:>10.2f}±{ms([r[name]['recall'] for r in by_stealth[s]])[1]:.2f}" for s in STEALTH))
    print("\nPR-AUC:")
    for name in MODELS:
        print(f"{name:<40}" + "".join(f"{ms([r[name]['pr_auc'] for r in by_stealth[s]])[0]:>16.2f}" for s in STEALTH))
    print("\nPrecision of the top 300 (max possible ~0.67):", {n: [round(ms([r[n]['precision'] for r in by_stealth[s]])[0], 2) for s in STEALTH] for n in ("account (A)", "account + flow + network")})
    print("Rules precision:", [round(ms([r['rules']['precision'] for r in by_stealth[s]])[0], 3) for s in STEALTH])
    print("Recall at the rules' own alert volume:", {n: [round(ms([r[n]['recall_at_rules_volume'] for r in by_stealth[s]])[0], 2) for s in STEALTH] for n in ("account (A)", "account + flow + network")})

    for s in (0.0, 1.0):
        print(f"\nRecall by typology at stealth {s}:")
        for m in ("rules", "account (A)", "account + flow match", "account + network (A+B)", "account + flow + network"):
            print(f"  {m:<26}", {t: round(float(np.mean([r[m]['recall_by_typology'][t] for r in by_stealth[s]])), 2) for t in TYPOLOGIES})
    print("\nPaired recall gain of adding network features to account features, per repetition:")
    for s in STEALTH:
        print(f"  stealth {s}: A+B vs A", [round(r['account + network (A+B)']['recall'] - r['account (A)']['recall'], 3) for r in by_stealth[s]],
              "| flow+B vs flow", [round(r['account + flow + network']['recall'] - r['account + flow match']['recall'], 3) for r in by_stealth[s]])  # fmt: skip
    print("\nFalse positives in the queue at stealth 0.5 (summed over repetitions):")
    for m in ("rules", "account (A)", "account + flow + network"):
        agg: dict = {}
        for r in by_stealth[0.5]:
            for k, v in r[m]["false_positives"].items():
                agg[k] = agg.get(k, 0) + v
        print(f"  {m:<26}", dict(sorted(agg.items(), key=lambda kv: -kv[1])))

    cov = {c: [run_rep(rep, coverage=c, seed=i) for i, rep in enumerate(reps[0.5])] for c in (1.0, 0.6, 0.3)}
    print("\nLabel coverage (recall in top 300, stealth 0.5):")
    for c in cov:
        print(f"  {c:.0%}", {n: round(float(np.mean([r[n]['recall'] for r in cov[c]])), 2) for n in ("account (A)", "account + flow + network")})

    plot_stealth(by_stealth)
    plot_typology(by_stealth)
    plot_coverage(cov)
    strip = lambda r: {k: v for k, v in r.items() if k != "_scores"}
    (ROOT / "reports" / "results.json").write_text(json.dumps(
        {"stealth": {str(s): [strip(r) for r in v] for s, v in by_stealth.items()}, "coverage": {str(c): [strip(r) for r in v] for c, v in cov.items()}}, indent=2))


if __name__ == "__main__":
    main()
