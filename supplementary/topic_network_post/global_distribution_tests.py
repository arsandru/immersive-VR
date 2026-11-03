#!/usr/bin/env python3
"""Global tests: are topics distributed differently across conditions?

No central topic is singled out. Participants are the unit (labels are permuted
across participants, group sizes fixed), so repeated sentences from one person
do not inflate significance.

A. Topic composition (how much each condition talks about each topic)
   * pooled-sentence generalised Jensen-Shannon divergence (GJSD, bits) between
     the three conditions' topic distributions;
   * PERMANOVA pseudo-F on participants' topic-proportion vectors (Hellinger).
   Per-topic contributions to the GJSD show which topics drive a difference.

B. Transition structure, per flow scope (where does each topic lead next?)
   * source-weighted GJSD of the conditions' outgoing-transition distributions,
     T = sum_i (n_i / n) * GJSD_i, conditioning on the source topic i, so it is
     independent of how often each topic is mentioned (per-source
     contributions reported);
   * GJSD between the conditions' stationary (PageRank, smoothed) distributions.

Pairwise versions of the GJSD tests use Holm correction over the 3 pairs.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_network_analysis import (  # noqa: E402
    CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, SCOPES, bh_fdr, build_participants,
    labels_for, load_data,
)
from pagerank_within_condition_tests import counts_from, pagerank, transition_matrix  # noqa: E402

PAIRS = list(combinations(range(3), 2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--smooth", type=float, default=0.1,
                   help="pseudo-count per off-diagonal cell for the stationary distribution")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def xlogx(x):
    out = np.zeros_like(x, dtype=float)
    nz = x > 0
    out[nz] = x[nz] * np.log2(x[nz])
    return out


def gjsd_terms(P, w):
    """Per-category contributions to the generalised JSD (rows of P = distributions)."""
    m = (w[:, None] * P).sum(axis=0)
    return -xlogx(m) + (w[:, None] * xlogx(P)).sum(axis=0)


def composition_stats(counts):
    """counts: (3, K) pooled sentence counts -> total GJSD, per-topic terms."""
    n = counts.sum(axis=1)
    P = counts / n[:, None]
    terms = gjsd_terms(P, n / n.sum())
    return terms.sum(), terms


def transition_stats(Cg):
    """Cg: (3, K, K) counts -> weighted GJSD total, per-source contributions."""
    n_i = Cg.sum(axis=2)  # (3, K) outgoing counts per condition and source
    tot_i = n_i.sum(axis=0)
    contrib = np.zeros(Cg.shape[1])
    for i in np.nonzero(tot_i)[0]:
        used = n_i[:, i] > 0
        P = Cg[used, i, :] / n_i[used, i][:, None]
        contrib[i] = gjsd_terms(P, n_i[used, i] / tot_i[i]).sum() * tot_i[i]
    contrib /= tot_i.sum()
    return contrib.sum(), contrib


def stationary_stat(Cg, eps):
    Pr = []
    for c in Cg:
        P = transition_matrix(c + eps * (1 - np.eye(len(c))))
        Pr.append(pagerank(P))
    Pr = np.array(Pr)
    return gjsd_terms(Pr, np.full(3, 1 / 3)).sum()


def permanova_F(X, groups):
    N, g = len(X), len(groups)
    grand = X.mean(axis=0)
    ssb = ssw = 0.0
    for idx in groups:
        mu = X[idx].mean(axis=0)
        ssb += len(idx) * ((mu - grand) ** 2).sum()
        ssw += ((X[idx] - mu) ** 2).sum()
    return (ssb / (g - 1)) / (ssw / (N - g))


def holm(p):
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty(len(p))
    out[order] = np.clip(adj, 0, 1)
    return out


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    df = load_data(args.input, include_outlier=False)
    topics = sorted(df["Topic"].unique().tolist())
    K = len(topics)
    labels = labels_for(topics)
    people = build_participants(df, np.zeros((len(df), 1)), topics)
    pids = [p for c in CONDITIONS for p in
            [q for q, v in people.items() if v["condition"] == c]]
    sizes = [sum(people[p]["condition"] == c for p in pids) for c in CONDITIONS]
    N = len(pids)

    # per-participant tables
    counts_p = np.stack([np.bincount(people[p]["topic_idx"], minlength=K) for p in pids]).astype(float)
    props = counts_p / counts_p.sum(axis=1, keepdims=True)
    hell = np.sqrt(props)
    trans_p = {}
    for scope in SCOPES:
        mats = []
        for p in pids:
            arrs = [np.asarray(s) for s in people[p][scope]]
            arr = np.concatenate(arrs)
            seg = np.concatenate([np.full(len(a), i) for i, a in enumerate(arrs)])
            mats.append(counts_from(arr, seg, K))
        trans_p[scope] = np.stack(mats)

    def groups_from(order):
        out_, s = [], 0
        for n in sizes:
            out_.append(np.asarray(order[s:s + n]))
            s += n
        return out_

    def all_stats(groups):
        """dict name -> (statistic, per-category vector or None)."""
        res = {}
        cg = np.stack([counts_p[g].sum(axis=0) for g in groups])
        tot, terms = composition_stats(cg)
        res["composition_gjsd"] = (tot, terms)
        for (i, j) in PAIRS:
            res[f"composition_gjsd|{i}-{j}"] = (composition_stats(cg[[i, j]])[0], None)
            res[f"composition_permanova_F|{i}-{j}"] = (permanova_F(hell, [groups[i], groups[j]]), None)
        res["composition_permanova_F"] = (permanova_F(hell, groups), None)
        for scope in SCOPES:
            Cg = np.stack([trans_p[scope][g].sum(axis=0) for g in groups])
            tot, contrib = transition_stats(Cg)
            res[f"transition_gjsd|{scope}"] = (tot, contrib)
            for (i, j) in PAIRS:
                res[f"transition_gjsd|{scope}|{i}-{j}"] = (transition_stats(Cg[[i, j]])[0], None)
            res[f"stationary_gjsd|{scope}"] = (stationary_stat(Cg, args.smooth), None)
        return res

    obs_groups = groups_from(list(range(N)))
    obs = all_stats(obs_groups)
    null = {k: [] for k in obs}
    null_vec = {k: [] for k, v in obs.items() if v[1] is not None}
    for n in range(args.n_perm):
        res = all_stats(groups_from(list(rng.permutation(N))))
        for k, (s, v) in res.items():
            null[k].append(s)
            if v is not None:
                null_vec[k].append(v)
        if (n + 1) % 1000 == 0:
            print(f"  permutation {n + 1}/{args.n_perm}", flush=True)
    null = {k: np.array(v) for k, v in null.items()}
    null_vec = {k: np.array(v) for k, v in null_vec.items()}
    pval = lambda o, nl: (1 + np.sum(nl >= o - 1e-12)) / (1 + len(nl))  # noqa: E731

    rows = []
    names = {"composition_gjsd": ("A. Topic composition", "all sentences", "GJSD (bits)"),
             "composition_permanova_F": ("A. Topic composition", "participants", "PERMANOVA pseudo-F")}
    for k in ("composition_gjsd", "composition_permanova_F"):
        fam, unit, stat = names[k]
        rows.append({"family": fam, "scope": "all", "test": stat, "comparison": "all three",
                     "statistic": obs[k][0], "null_mean": null[k].mean(),
                     "null_95": np.percentile(null[k], 95), "p_perm": pval(obs[k][0], null[k])})
    for scope in SCOPES:
        for k, stat, fam in ((f"transition_gjsd|{scope}", "source-weighted GJSD (bits)", "B. Transition structure"),
                             (f"stationary_gjsd|{scope}", "GJSD of PageRank vectors (bits)", "B. Stationary distribution")):
            rows.append({"family": fam, "scope": scope, "test": stat, "comparison": "all three",
                         "statistic": obs[k][0], "null_mean": null[k].mean(),
                         "null_95": np.percentile(null[k], 95), "p_perm": pval(obs[k][0], null[k])})
    pair_rows = []
    for (i, j) in PAIRS:
        cmp_ = f"{CONDITIONS[i]} vs {CONDITIONS[j]}"
        for k, fam, scope, stat in ((f"composition_gjsd|{i}-{j}", "A. Topic composition", "all", "GJSD (bits)"),
                                    (f"composition_permanova_F|{i}-{j}", "A. Topic composition", "all", "PERMANOVA pseudo-F")):
            pair_rows.append({"family": fam, "scope": scope, "test": stat, "comparison": cmp_,
                              "statistic": obs[k][0], "null_mean": null[k].mean(),
                              "null_95": np.percentile(null[k], 95), "p_perm": pval(obs[k][0], null[k])})
        for scope in SCOPES:
            k = f"transition_gjsd|{scope}|{i}-{j}"
            pair_rows.append({"family": "B. Transition structure", "scope": scope,
                              "test": "source-weighted GJSD (bits)", "comparison": cmp_,
                              "statistic": obs[k][0], "null_mean": null[k].mean(),
                              "null_95": np.percentile(null[k], 95), "p_perm": pval(obs[k][0], null[k])})
    pair = pd.DataFrame(pair_rows)
    pair["p_holm"] = pair.groupby(["family", "scope", "test"])["p_perm"].transform(
        lambda s: holm(s.to_numpy()))
    tests = pd.concat([pd.DataFrame(rows), pair], ignore_index=True)
    tests.to_csv(out / "global_distribution_tests.csv", index=False)

    crow = []
    for k, fam, scope in [("composition_gjsd", "topic (composition)", "all")] + \
            [(f"transition_gjsd|{s}", "source topic (transitions)", s) for s in SCOPES]:
        pv = np.array([(1 + np.sum(null_vec[k][:, t] >= obs[k][1][t] - 1e-12)) / (1 + args.n_perm)
                       for t in range(K)])
        fdr = bh_fdr(pv)
        for t in range(K):
            crow.append({"family": fam, "scope": scope, "Topic": topics[t],
                         "topic_label": labels[topics[t]], "contribution": obs[k][1][t],
                         "null_mean": null_vec[k][:, t].mean(),
                         "null_low": np.percentile(null_vec[k][:, t], 2.5),
                         "null_high": np.percentile(null_vec[k][:, t], 97.5),
                         "p_perm": pv[t], "p_fdr": fdr[t]})
    pd.DataFrame(crow).to_csv(out / "global_distribution_contributions.csv", index=False)
    np.savez(out / "global_distribution_null.npz", **{k.replace("|", "__"): v for k, v in null.items()},
             **{"obs__" + k.replace("|", "__"): np.array(obs[k][0]) for k in obs})
    print(tests.round(4).to_string())
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
