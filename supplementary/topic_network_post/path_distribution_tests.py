#!/usr/bin/env python3
"""Between-condition tests of the probabilistic paths leading to a central topic.

For every flow scope, the targets are the topics that are most central (highest
PageRank) in at least one condition. For each target and each condition:

* Transition probabilities P_ij come from the condition's flow graph, with a
  small additive smoothing (``--smooth``, pseudo-count per off-diagonal cell)
  so topics with few observed exits do not look deterministic.
* Paths start from the observed distribution of opening topics (sequences that
  already open on the target are excluded) and end at the *first* arrival at
  the target, at most ``--max-steps`` steps later. A path's probability is the
  product of P along it (as in the paper's PSBC).
* Reach probability = total probability of such paths. Entry routes = the same
  mass split by the last topic visited before the target.

Conditions are compared with a participant-label permutation test: reach
probability and per-route entry mass (two-sided, BH over routes) and the
Jensen-Shannon distance between the routes' normalised distributions.
Participant-bootstrap 95% CIs for the per-condition values.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_network_analysis import (  # noqa: E402
    CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, SCOPES, bh_fdr, build_participants,
    labels_for, load_data,
)
from pagerank_within_condition_tests import counts_from  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--smooth", type=float, default=0.1,
                   help="main smoothing; 0.02 and 0.5 are run as sensitivity checks")
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pack(pids, arrays, scope):
    """Concatenate the sequences of a participant group -> (topics, segment ids)."""
    arrs, segs, s = [], [], 0
    for pid in pids:
        for a in arrays[pid][scope]:
            arrs.append(a)
            segs.append(np.full(len(a), s))
            s += 1
    return np.concatenate(arrs), np.concatenate(segs)


def group_inputs(pids, arrays, scope, K):
    arr, seg = pack(pids, arrays, scope)
    C = counts_from(arr, seg, K)
    starts = np.r_[True, seg[1:] != seg[:-1]]
    first = np.bincount(arr[starts], minlength=K).astype(float)
    return C, first


def entry_mass(C, first, t, eps, steps):
    """Probability of first arriving at t within `steps`, split by last stop."""
    K = len(C)
    P = C + eps * (1 - np.eye(K))
    P = P / P.sum(axis=1, keepdims=True)
    m = first.copy()
    m[t] = 0
    if m.sum() == 0:  # every sequence opens on the target (only in resamples)
        return np.full(K, np.nan), P
    m = m / m.sum()
    Qz = P.copy()
    Qz[:, t] = 0
    Qz[t, :] = 0
    e = np.zeros(K)
    for _ in range(steps):
        e += m * P[:, t]
        m = m @ Qz
    return e, P


def top_paths(P, first, t, steps, n=5):
    K = len(P)
    pi0 = first.copy()
    pi0[t] = 0
    pi0 = pi0 / pi0.sum()
    found = []

    def walk(path, prob):
        i = path[-1]
        if prob <= 0:
            return
        if P[i, t] > 0:
            found.append((path + [t], prob * P[i, t]))
        if len(path) < steps:
            for j in range(K):
                if j != t and j != i:
                    walk(path + [j], prob * P[i, j])

    for i0 in range(K):
        if i0 != t:
            walk([i0], pi0[i0])
    return sorted(found, key=lambda x: -x[1])[:n]


def summarise(e):
    reach = e.sum()
    return reach, (e / reach if reach > 0 else e)


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    smooths = sorted({0.02, args.smooth, 0.5})
    df = load_data(args.input, include_outlier=False)
    topics = sorted(df["Topic"].unique().tolist())
    K = len(topics)
    labels = labels_for(topics)
    idx = {t: i for i, t in enumerate(topics)}
    people = build_participants(df, np.zeros((len(df), 1)), topics)
    arrays = {p: {s: [np.asarray(q) for q in v[s]] for s in SCOPES} for p, v in people.items()}
    by_cond = {c: [p for p, v in people.items() if v["condition"] == c] for c in CONDITIONS}
    all_pids = [p for c in CONDITIONS for p in by_cond[c]]
    sizes = [len(by_cond[c]) for c in CONDITIONS]

    tops = pd.read_csv(out / "pagerank_within_condition_topic_tests.csv")
    tops = tops[tops.is_top]
    targets = {s: [idx[t] for t in dict.fromkeys(tops[tops.scope == s]["Topic"])]
               for s in SCOPES}
    print({s: [labels[topics[t]] for t in v] for s, v in targets.items()})
    pairs = list(combinations(range(3), 2))

    def evaluate(groups):
        """{(scope, target, eps): (reach[3], e[3, K])} for three participant groups."""
        res = {}
        for scope in SCOPES:
            gi = [group_inputs(g, arrays, scope, K) for g in groups]
            for t in targets[scope]:
                for eps in smooths:
                    es = [entry_mass(C, f, t, eps, args.max_steps)[0] for C, f in gi]
                    res[(scope, t, eps)] = (np.array([e.sum() for e in es]), np.stack(es))
        return res

    def stat_vectors(reach, e):
        """diffs for reach (3 pairs), entry mass (3 pairs x K), JSD (3 pairs)."""
        d_reach = np.array([reach[i] - reach[j] for i, j in pairs])
        d_e = np.stack([e[i] - e[j] for i, j in pairs])
        r = [summarise(e[i])[1] for i in range(3)]
        jsd = np.array([jensenshannon(r[i], r[j], base=2) for i, j in pairs])
        return d_reach, d_e, jsd

    obs_groups = [by_cond[c] for c in CONDITIONS]
    obs = evaluate(obs_groups)

    # ---- bootstrap CIs (main smoothing only)
    boot = {k: [] for k in obs if k[2] == args.smooth}
    for _ in range(args.n_boot):
        groups = [list(rng.choice(by_cond[c], size=len(by_cond[c]), replace=True))
                  for c in CONDITIONS]
        res = evaluate(groups)
        for k in boot:
            boot[k].append(res[k])

    # ---- permutation null
    exceed = {k: [np.zeros(3), np.zeros((3, K)), np.zeros(3)] for k in obs}
    obs_stats = {k: stat_vectors(*obs[k]) for k in obs}
    for n in range(args.n_perm):
        perm = list(rng.permutation(all_pids))
        groups, s0 = [], 0
        for s in sizes:
            groups.append(perm[s0:s0 + s])
            s0 += s
        res = evaluate(groups)
        for k in obs:
            d_reach, d_e, jsd = stat_vectors(*res[k])
            o = obs_stats[k]
            exceed[k][0] += np.abs(d_reach) >= np.abs(o[0]) - 1e-12
            exceed[k][1] += np.abs(d_e) >= np.abs(o[1]) - 1e-12
            exceed[k][2] += jsd >= o[2] - 1e-12
        if (n + 1) % 250 == 0:
            print(f"  permutation {n + 1}/{args.n_perm}", flush=True)

    # ---- tables
    reach_rows, route_rows, test_rows, path_rows = [], [], [], []
    for (scope, t, eps), (reach, e) in obs.items():
        tl = labels[topics[t]]
        for ci, cond in enumerate(CONDITIONS):
            row = {"scope": scope, "target_topic": topics[t], "target_label": tl,
                   "condition": cond, "smooth": eps, "reach_prob": reach[ci],
                   "ci_low": np.nan, "ci_high": np.nan}
            if (scope, t, eps) in boot:
                b = np.array([x[0][ci] for x in boot[(scope, t, eps)]])
                row["ci_low"], row["ci_high"] = np.nanpercentile(b, [2.5, 97.5])
            reach_rows.append(row)
            C, first = group_inputs(by_cond[cond], arrays, scope, K)
            if eps == args.smooth:
                for path, pr in top_paths(entry_mass(C, first, t, eps, args.max_steps)[1],
                                          first, t, args.max_steps):
                    path_rows.append({"scope": scope, "target_label": tl, "condition": cond,
                                      "path": " > ".join(labels[topics[i]] for i in path),
                                      "n_steps": len(path) - 1, "probability": pr})
            for j in range(K):
                if j == t:
                    continue
                rr = {"scope": scope, "target_topic": topics[t], "target_label": tl,
                      "condition": cond, "smooth": eps, "source_topic": topics[j],
                      "source_label": labels[topics[j]], "entry_mass": e[ci, j],
                      "entry_share": e[ci, j] / reach[ci] if reach[ci] > 0 else np.nan,
                      "n_observed_transitions": C[j, t], "ci_low": np.nan, "ci_high": np.nan}
                if (scope, t, eps) in boot:
                    b = np.array([x[1][ci, j] for x in boot[(scope, t, eps)]])
                    rr["ci_low"], rr["ci_high"] = np.nanpercentile(b, [2.5, 97.5])
                route_rows.append(rr)
        d_reach, d_e, jsd = obs_stats[(scope, t, eps)]
        ex = exceed[(scope, t, eps)]
        base = {"scope": scope, "target_topic": topics[t], "target_label": tl, "smooth": eps}
        for pi_, (i, j) in enumerate(pairs):
            ab = {"condition_a": CONDITIONS[i], "condition_b": CONDITIONS[j]}
            test_rows.append({**base, **ab, "test": "reach_probability", "source_label": "",
                              "value_a": reach[i], "value_b": reach[j], "difference": d_reach[pi_],
                              "p_perm": (1 + ex[0][pi_]) / (1 + args.n_perm)})
            test_rows.append({**base, **ab, "test": "route_distribution_jsd", "source_label": "",
                              "value_a": np.nan, "value_b": np.nan, "difference": jsd[pi_],
                              "p_perm": (1 + ex[2][pi_]) / (1 + args.n_perm)})
            for j2 in range(K):
                if j2 == t:
                    continue
                test_rows.append({**base, **ab, "test": "entry_mass", "source_label": labels[topics[j2]],
                                  "value_a": e[i, j2], "value_b": e[j, j2], "difference": d_e[pi_, j2],
                                  "p_perm": (1 + ex[1][pi_, j2]) / (1 + args.n_perm)})
    tests = pd.DataFrame(test_rows)
    tests["p_fdr"] = tests.groupby(["scope", "target_topic", "smooth", "test", "condition_a",
                                    "condition_b"])["p_perm"].transform(lambda s: bh_fdr(s.to_numpy()))
    tests.to_csv(out / "path_condition_tests.csv", index=False)
    pd.DataFrame(reach_rows).to_csv(out / "path_reach_by_condition.csv", index=False)
    pd.DataFrame(route_rows).to_csv(out / "path_entry_routes_by_condition.csv", index=False)
    pd.DataFrame(path_rows).to_csv(out / "path_top_paths.csv", index=False)
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
