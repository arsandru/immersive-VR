#!/usr/bin/env python3
"""Jensen-Shannon test on the path distributions implied by the condition graphs.

Each condition's flow graph is a Markov chain (transition probabilities P_ij,
additive smoothing ``--smooth`` per off-diagonal cell). It assigns a probability
to every path of L steps:  Prob(i0 -> i1 -> ... -> iL) = pi0(i0) * prod P.
The JSD (bits) between two conditions' distributions over all such paths
measures how differently their graphs route the walk; the omnibus statistic is
the generalised JSD of the three distributions.

* start distribution: "pooled" = opening-topic distribution of all participants
  (identical for every condition, so only the graphs differ), or "own" = each
  condition's own openings.
* Null: participants' condition labels are permuted (group sizes fixed).
* Pairwise p-values use Holm correction over the three pairs.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_network_analysis import (  # noqa: E402
    CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, SCOPES, build_participants, labels_for,
    load_data,
)
from pagerank_within_condition_tests import counts_from  # noqa: E402
from global_distribution_tests import gjsd_terms, holm  # noqa: E402

PAIRS = list(combinations(range(3), 2))
LENGTHS = (2, 3)
STARTS = ("pooled", "own")
MAIN = {"L": 3, "start": "pooled", "smooth": 0.1}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--smooth", type=float, default=0.1,
                   help="main smoothing; 0.02 and 0.5 are run as sensitivity checks")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chain(C, eps):
    K = len(C)
    P = C + eps * (1 - np.eye(K))
    return P / P.sum(axis=1, keepdims=True)


def path_probs(P, pi0, L):
    """Flat vector of probabilities of all L-step paths (K^(L+1) entries)."""
    p = pi0
    for _ in range(L):
        p = p[..., None] * P
    return p.reshape(-1)


def jsd_pair(p, q):
    return gjsd_terms(np.stack([p, q]), np.array([0.5, 0.5])).sum()


def main() -> None:
    args = parse_args()
    MAIN["smooth"] = args.smooth
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    smooths = sorted({0.02, args.smooth, 0.5})
    df = load_data(args.input, include_outlier=False)
    topics = sorted(df["Topic"].unique().tolist())
    K = len(topics)
    labels = labels_for(topics)
    people = build_participants(df, np.zeros((len(df), 1)), topics)
    pids = [p for c in CONDITIONS for p in [q for q, v in people.items() if v["condition"] == c]]
    sizes = [sum(people[p]["condition"] == c for p in pids) for c in CONDITIONS]
    N = len(pids)

    Cp, Fp = {}, {}
    for scope in SCOPES:
        mats, firsts = [], []
        for p in pids:
            arrs = [np.asarray(s) for s in people[p][scope]]
            arr = np.concatenate(arrs)
            seg = np.concatenate([np.full(len(a), i) for i, a in enumerate(arrs)])
            mats.append(counts_from(arr, seg, K))
            starts = np.r_[True, seg[1:] != seg[:-1]]
            firsts.append(np.bincount(arr[starts], minlength=K))
        Cp[scope], Fp[scope] = np.stack(mats), np.stack(firsts).astype(float)
    pooled_start = {s: Fp[s].sum(axis=0) / Fp[s].sum() for s in SCOPES}
    configs = [(s, L, st, eps) for s in SCOPES for L in LENGTHS for st in STARTS for eps in smooths]

    def groups_from(order):
        res, s = [], 0
        for n in sizes:
            res.append(np.asarray(order[s:s + n]))
            s += n
        return res

    def evaluate(groups):
        """{config: (omnibus GJSD, pairwise JSD[3], path distributions)}"""
        res = {}
        for scope in SCOPES:
            Cg = [Cp[scope][g].sum(axis=0) for g in groups]
            Fg = [Fp[scope][g].sum(axis=0) for g in groups]
            for eps in smooths:
                Ps = [chain(C, eps) for C in Cg]
                for st in STARTS:
                    pis = [pooled_start[scope]] * 3 if st == "pooled" else [f / f.sum() for f in Fg]
                    for L in LENGTHS:
                        dists = np.stack([path_probs(P, pi, L) for P, pi in zip(Ps, pis)])
                        omni = gjsd_terms(dists, np.full(3, 1 / 3)).sum()
                        pair = np.array([jsd_pair(dists[i], dists[j]) for i, j in PAIRS])
                        res[(scope, L, st, eps)] = (omni, pair, dists)
        return res

    obs = evaluate(groups_from(list(range(N))))
    exceed = {c: [0, np.zeros(3)] for c in configs}
    nulls = {c: [] for c in configs if c[1:] == (MAIN["L"], MAIN["start"], MAIN["smooth"])}
    for n in range(args.n_perm):
        res = evaluate(groups_from(list(rng.permutation(N))))
        for c in configs:
            o, r = obs[c], res[c]
            exceed[c][0] += r[0] >= o[0] - 1e-12
            exceed[c][1] += r[1] >= o[1] - 1e-12
            if c in nulls:
                nulls[c].append((r[0], *r[1]))
        if (n + 1) % 1000 == 0:
            print(f"  permutation {n + 1}/{args.n_perm}", flush=True)

    rows = []
    for c in configs:
        scope, L, st, eps = c
        omni, pair, _ = obs[c]
        base = {"scope": scope, "path_length": L, "start": st, "smooth": eps}
        rows.append({**base, "comparison": "all three", "jsd_bits": omni,
                     "p_perm": (1 + exceed[c][0]) / (1 + args.n_perm), "p_holm": np.nan})
        p = (1 + exceed[c][1]) / (1 + args.n_perm)
        ph = holm(p)
        for k, (i, j) in enumerate(PAIRS):
            rows.append({**base, "comparison": f"{CONDITIONS[i]} vs {CONDITIONS[j]}",
                         "jsd_bits": pair[k], "p_perm": p[k], "p_holm": ph[k]})
    tests = pd.DataFrame(rows)
    tests.to_csv(out / "path_graph_jsd_tests.csv", index=False)
    np.savez(out / "path_graph_jsd_null.npz",
             **{f"{c[0]}": np.array(v) for c, v in nulls.items()},
             **{f"obs_{c[0]}": np.array([obs[c][0], *obs[c][1]]) for c in nulls})

    # paths driving the divergence (main setting): contribution to the omnibus GJSD
    prow = []
    for scope in SCOPES:
        L = MAIN["L"]
        dists = obs[(scope, L, MAIN["start"], MAIN["smooth"])][2]
        terms = gjsd_terms(dists, np.full(3, 1 / 3))
        for flat in np.argsort(-terms)[:15]:
            path = np.unravel_index(flat, (K,) * (L + 1))
            prow.append({"scope": scope, "path": " > ".join(labels[topics[i]] for i in path),
                         "contribution_bits": terms[flat], "share_of_total": terms[flat] / terms.sum(),
                         **{f"prob_{cnd}": dists[ci][flat] for ci, cnd in enumerate(CONDITIONS)}})
    pd.DataFrame(prow).to_csv(out / "path_graph_jsd_top_paths.csv", index=False)
    show = tests[(tests.path_length == MAIN["L"]) & (tests.start == MAIN["start"]) &
                 (tests.smooth == args.smooth)]
    print(show.round(4).to_string())
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
