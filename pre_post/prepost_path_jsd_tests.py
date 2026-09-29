#!/usr/bin/env python3
"""Jensen-Shannon tests on the path distributions of the topic flow graphs, pre vs post, within and between conditions.

Each (condition, timepoint) flow graph is a Markov chain over the 11 topics (transition probabilities from the observed
consecutive-descriptor transitions of participants' Q1 lists, additive smoothing ``--smooth`` per off-diagonal cell). With one
shared opening-topic distribution (pooled over all participants and both timepoints, so only the graphs differ) it assigns a
probability to every L-step path (default L = 3). All statistics are JSD in bits between such path distributions.

WITHIN condition  J_c = JSD(pre graph of c, post graph of c), paired participants. Null: pre/post labels swapped independently
                  within participant (random sign flip); one-sided p (large J = large change).
BETWEEN conditions at one timepoint (pre; post): JSD between two conditions' graphs, plus the generalised JSD of all three.
                  Null: condition labels permuted across the participants observed at that timepoint.
INTERACTION       does the pre->post change differ between conditions? (a) |J_c1 - J_c2|: difference in the AMOUNT of change;
                  (b) TV distance between the two conditions' signed change vectors (post - pre path distributions): difference
                  in the DIRECTION of change. Null: condition labels permuted across the paired participants.
Adjustment: BH-FDR within each family (three conditions; three pairs per test).
Sensitivity: path length 2, smoothing 0.02 and 0.5 (written to the same CSV), and the ``--drop-review`` subset.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from prepost_common import (CONDITIONS, OUT, TOPIC_LABELS, bh_fdr, counts_from, gjsd_terms, load_units, participants)

K = [t for t in sorted(TOPIC_LABELS) if t >= 0]
n = len(K)
kidx = {t: i for i, t in enumerate(K)}
PAIRS = list(combinations(range(3), 2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--smooth", type=float, default=0.1)
    p.add_argument("--path-length", type=int, default=3)
    p.add_argument("--drop-review", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chain(C, eps):
    P = C + eps * (1 - np.eye(n))
    return P / P.sum(axis=1, keepdims=True)


def path_dist(P, pi0, L):
    p = pi0
    for _ in range(L):
        p = p[..., None] * P
    return p.reshape(-1)


def jsd(p, q):
    return gjsd_terms(np.stack([p, q]), np.array([0.5, 0.5])).sum()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    u = load_units(drop_review=args.drop_review)
    P = participants(u)
    pids = P.participant_id.tolist()
    ix = {p: i for i, p in enumerate(pids)}
    Cm = np.zeros((len(pids), 2, n, n))
    F0 = np.zeros((len(pids), 2, n))
    for (pid, time), g in u.groupby(["participant_id", "time"]):
        arr = np.array([kidx[t] for t in g.Topic])
        s = 0 if time == "pre" else 1
        Cm[ix[pid], s] = counts_from(arr, np.zeros(len(arr), int), n)
        F0[ix[pid], s, arr[0]] = 1
    cond = P.cond.to_numpy()
    has = np.stack([P.has_pre.to_numpy(), P.has_post.to_numpy()], axis=1)
    both = has.all(axis=1)
    pi0 = F0.sum(axis=(0, 1)) / F0.sum()
    configs = sorted({(args.path_length, args.smooth), (2, args.smooth), (args.path_length, 0.02), (args.path_length, 0.5)})

    def dists(C_pair, cfg):
        L, eps = cfg
        return [path_dist(chain(C, eps), pi0, L) for C in C_pair]

    rows = []

    def add(cfg, analysis, test, comparison, n_units, obs, null, larger_is_more=True):
        null = np.asarray(null)
        rows.append({"path_length": cfg[0], "smooth": cfg[1], "analysis": analysis, "test": test, "comparison": comparison,
                     "n": n_units, "statistic": obs, "null_mean": null.mean(), "null_95": np.percentile(null, 95),
                     "p_perm": (1 + np.sum(null >= obs - 1e-12)) / (1 + len(null))})

    # ---------------- within condition (sign flip on paired participants)
    for cfg in configs:
        for c, cname in enumerate(CONDITIONS):
            idx = np.nonzero((cond == c) & both)[0]

            def J(swap):
                pre = sum(Cm[i, 1 if s else 0] for i, s in zip(idx, swap))
                post = sum(Cm[i, 0 if s else 1] for i, s in zip(idx, swap))
                d = dists([pre, post], cfg)
                return jsd(d[0], d[1])

            obs = J(np.zeros(len(idx), bool))
            null = [J(rng.random(len(idx)) < 0.5) for _ in range(args.n_perm)]
            add(cfg, "within condition (pre vs post)", "JSD of path distributions", cname, len(idx), obs, null)

    # ---------------- between conditions at one timepoint (label permutation)
    for cfg in configs:
        for s, tname in enumerate(("pre", "post")):
            idx = np.nonzero(has[:, s])[0]
            lab0 = cond[idx]

            def stats(lab):
                d = dists([Cm[idx[lab == c], s].sum(axis=0) for c in range(3)], cfg)
                pair = np.array([jsd(d[i], d[j]) for i, j in PAIRS])
                omni = gjsd_terms(np.stack(d), np.full(3, 1 / 3)).sum()
                return omni, pair

            o_omni, o_pair = stats(lab0)
            null_omni, null_pair = [], []
            for _ in range(args.n_perm):
                a, b = stats(rng.permutation(lab0))
                null_omni.append(a)
                null_pair.append(b)
            null_pair = np.array(null_pair)
            an = f"between conditions ({tname})"
            add(cfg, an, "JSD of path distributions", "all three", len(idx), o_omni, null_omni)
            for k, (i, j) in enumerate(PAIRS):
                add(cfg, an, "JSD of path distributions", f"{CONDITIONS[i]} vs {CONDITIONS[j]}", int(np.isin(lab0, (i, j)).sum()),
                    o_pair[k], null_pair[:, k])

    # ---------------- interaction: does the pre -> post change differ by condition?
    idxb = np.nonzero(both)[0]
    lab0 = cond[idxb]
    for cfg in configs:
        def change_stats(lab):
            J, D = [], []
            for c in range(3):
                m = idxb[lab == c]
                d = dists([Cm[m, 0].sum(axis=0), Cm[m, 1].sum(axis=0)], cfg)
                J.append(jsd(d[0], d[1]))
                D.append(d[1] - d[0])
            amount = np.array([abs(J[i] - J[j]) for i, j in PAIRS])
            direction = np.array([0.5 * np.abs(D[i] - D[j]).sum() for i, j in PAIRS])
            return amount, direction

        oa, od = change_stats(lab0)
        na, nd = [], []
        for _ in range(args.n_perm):
            a, d_ = change_stats(rng.permutation(lab0))
            na.append(a)
            nd.append(d_)
        na, nd = np.array(na), np.array(nd)
        for k, (i, j) in enumerate(PAIRS):
            nn = int(np.isin(lab0, (i, j)).sum())
            add(cfg, "interaction (change differs by condition)", "difference in amount of change |J1 - J2|",
                f"{CONDITIONS[i]} vs {CONDITIONS[j]}", nn, oa[k], na[:, k])
            add(cfg, "interaction (change differs by condition)", "difference in direction of change (TV of change vectors)",
                f"{CONDITIONS[i]} vs {CONDITIONS[j]}", nn, od[k], nd[:, k])

    res = pd.DataFrame(rows)
    fam = ["path_length", "smooth", "analysis", "test"]
    res["p_fdr"] = np.nan
    m = res.comparison != "all three"
    res.loc[m, "p_fdr"] = res[m].groupby(fam)["p_perm"].transform(lambda s: bh_fdr(s.to_numpy()))
    res.to_csv(OUT / f"prepost_path_jsd_tests{args.tag}.csv", index=False)
    pd.set_option("display.width", 230)
    main = res[(res.path_length == args.path_length) & (res.smooth == args.smooth)]
    print(main[["analysis", "test", "comparison", "n", "statistic", "null_mean", "p_perm", "p_fdr"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
