#!/usr/bin/env python3
"""Pre vs post PageRank centrality on the topic flow graphs.

Flow graph per condition and timepoint: directed edge A->B whenever descriptor B follows
descriptor A inside one participant's Q1 list at that timepoint (order kept; self
transitions excluded; outlier units removed). Edge probability = transition probability;
PageRank uses damping 0.85, dangling topics jump uniformly (as in topic_network_post).

BASELINE (pre only): do the conditions' PageRank distributions differ before the
  intervention? Generalised JSD of the three PageRank vectors, participant labels permuted.
TIME (paired participants, per condition): change in PageRank per topic, post minus pre.
  Bootstrap 95% CI (participants resampled, both timepoints kept together); two-sided p
  from swapping pre/post within participants; BH over topics within condition.
  Also the JSD between the pre and post 3-step path distributions (shared opening-topic
  distribution, additive smoothing 0.1), same swap null.
INTERACTION: does the PageRank change differ by condition? Omnibus statistic = summed squared
  between-condition deviation of the topic changes; pairwise the same on two conditions
  (BH-FDR); per topic the same per topic (BH-FDR). Condition labels permuted across participants.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from prepost_common import (CONDITIONS, OUT, TOPIC_LABELS, bh_fdr, counts_from, gjsd_terms, label,
                            load_units, pagerank, participants, transition_matrix)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--smooth", type=float, default=0.1)
    p.add_argument("--path-length", type=int, default=3)
    p.add_argument("--drop-review", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chain(C, eps):
    K = len(C)
    P = C + eps * (1 - np.eye(K))
    return P / P.sum(axis=1, keepdims=True)


def path_dist(P, pi0, L):
    p = pi0
    for _ in range(L):
        p = p[..., None] * P
    return p.reshape(-1)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    u = load_units(drop_review=args.drop_review)
    P = participants(u)
    K = [t for t in sorted(TOPIC_LABELS) if t >= 0]
    n = len(K)
    kidx = {t: i for i, t in enumerate(K)}
    pids = P.participant_id.tolist()
    ix = {p: i for i, p in enumerate(pids)}
    Cm = np.zeros((len(pids), 2, n, n))
    F0 = np.zeros((len(pids), 2, n))
    for (pid, time), g in u.groupby(["participant_id", "time"]):
        arr = np.array([kidx[t] for t in g.Topic])
        Cm[ix[pid], 0 if time == "pre" else 1] = counts_from(arr, np.zeros(len(arr), int), n)
        F0[ix[pid], 0 if time == "pre" else 1, arr[0]] = 1
    cond = P.cond.to_numpy()
    has_pre, has_post = P.has_pre.to_numpy(), P.has_post.to_numpy()
    both = has_pre & has_post

    def pr_of(C):
        return pagerank(transition_matrix(C))

    def delta_pr(idx, swap=None):
        """PageRank(post) - PageRank(pre) for participant indices idx (optional swap mask)."""
        s = np.zeros(len(idx), int) if swap is None else swap.astype(int)
        pre = sum(Cm[i, s_] for i, s_ in zip(idx, s))
        post = sum(Cm[i, 1 - s_] for i, s_ in zip(idx, s))
        return pr_of(post) - pr_of(pre)

    rows_pr, rows_tests, rows_topic = [], [], []
    # ---------------- descriptive PageRank per condition x time (all available participants)
    for c, cname in enumerate(CONDITIONS):
        for t, tname in enumerate(("pre", "post")):
            idx = np.nonzero((cond == c) & (has_pre if t == 0 else has_post))[0]
            C = Cm[idx, t].sum(axis=0)
            pr = pr_of(C)
            boot = np.stack([pr_of(Cm[rng.choice(idx, len(idx)), t].sum(axis=0)) for _ in range(args.n_boot)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            for j, top in enumerate(K):
                rows_pr.append({"condition": cname, "time": tname, "Topic": top, "topic": label(top),
                                "pagerank": pr[j], "ci_low": lo[j], "ci_high": hi[j],
                                "n_participants": len(idx), "n_transitions": int(C.sum())})
    # ---------------- baseline
    idx_pre = np.nonzero(has_pre)[0]

    def base_stat(lab):
        vs = [pr_of(Cm[idx_pre[lab == c], 0].sum(axis=0)) for c in range(3)]
        return gjsd_terms(np.array(vs), np.full(3, 1 / 3)).sum()

    lab_pre = cond[idx_pre]
    obs = base_stat(lab_pre)
    ge = sum(base_stat(rng.permutation(lab_pre)) >= obs - 1e-12 for _ in range(args.n_perm))
    rows_tests.append({"analysis": "baseline (pre only)", "test": "GJSD of PageRank vectors", "comparison": "all three",
                       "n": len(idx_pre), "statistic": obs, "p_perm": (1 + ge) / (1 + args.n_perm)})

    # ---------------- time effect per condition (paired participants)
    def path_jsd(idx, swap=None):
        s = np.zeros(len(idx), int) if swap is None else swap.astype(int)
        pre = sum(Cm[i, s_] for i, s_ in zip(idx, s))
        post = sum(Cm[i, 1 - s_] for i, s_ in zip(idx, s))
        f = F0[idx, 0].sum(axis=0) + F0[idx, 1].sum(axis=0)
        pi0 = f / f.sum()
        d = np.stack([path_dist(chain(pre, args.smooth), pi0, args.path_length),
                      path_dist(chain(post, args.smooth), pi0, args.path_length)])
        return gjsd_terms(d, np.array([0.5, 0.5])).sum()

    deltas = {}
    for c, cname in enumerate(CONDITIONS):
        idx = np.nonzero((cond == c) & both)[0]
        obs = delta_pr(idx)
        deltas[c] = obs
        cnt = np.zeros(n)
        for _ in range(args.n_perm):
            cnt += np.abs(delta_pr(idx, rng.random(len(idx)) < 0.5)) >= np.abs(obs) - 1e-12
        p = (1 + cnt) / (1 + args.n_perm)
        adj = bh_fdr(p)
        boot = np.stack([delta_pr(rng.choice(idx, len(idx))) for _ in range(args.n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for j, top in enumerate(K):
            rows_topic.append({"condition": cname, "Topic": top, "topic": label(top), "pagerank_change": obs[j],
                               "ci_low": lo[j], "ci_high": hi[j], "p_perm": p[j], "p_fdr": adj[j], "n": len(idx)})
        o = path_jsd(idx)
        g = sum(path_jsd(idx, rng.random(len(idx)) < 0.5) >= o - 1e-12 for _ in range(args.n_perm // 5))
        rows_tests.append({"analysis": "pre vs post (paired)", "test": "JSD of 3-step path distributions",
                           "comparison": cname, "n": len(idx), "statistic": o, "p_perm": (1 + g) / (1 + args.n_perm // 5)})

    # ---------------- interaction (condition x time)
    idxb = np.nonzero(both)[0]
    labb = cond[idxb]

    def cond_deltas(lab):
        return np.stack([delta_pr(idxb[lab == c]) for c in range(3)])

    def stat_all(D, lab):
        w = np.array([(lab == c).sum() for c in range(3)])
        mu = (w[:, None] * D).sum(axis=0) / w.sum()
        return (w[:, None] * (D - mu) ** 2).sum(axis=0)  # per topic

    obs_D = cond_deltas(labb)
    obs_topic = stat_all(obs_D, labb)
    cnt_topic = np.zeros(n)
    cnt_all = 0
    pair_obs = {(i, j): ((obs_D[i] - obs_D[j]) ** 2).sum() for i, j in combinations(range(3), 2)}
    pair_cnt = {k: 0 for k in pair_obs}
    for _ in range(args.n_perm):
        lp = rng.permutation(labb)
        D = cond_deltas(lp)
        st = stat_all(D, lp)
        cnt_topic += st >= obs_topic - 1e-12
        cnt_all += st.sum() >= obs_topic.sum() - 1e-12
        for (i, j) in pair_obs:
            pair_cnt[(i, j)] += ((D[i] - D[j]) ** 2).sum() >= pair_obs[(i, j)] - 1e-12
    rows_tests.append({"analysis": "interaction (change differs by condition)", "test": "PageRank change, omnibus",
                       "comparison": "all three", "n": len(idxb), "statistic": obs_topic.sum(),
                       "p_perm": (1 + cnt_all) / (1 + args.n_perm)})
    for (i, j), o in pair_obs.items():
        rows_tests.append({"analysis": "interaction (change differs by condition)", "test": "PageRank change, pairwise",
                           "comparison": f"{CONDITIONS[i]} vs {CONDITIONS[j]}",
                           "n": int(np.isin(labb, (i, j)).sum()), "statistic": o,
                           "p_perm": (1 + pair_cnt[(i, j)]) / (1 + args.n_perm)})
    p_topic = (1 + cnt_topic) / (1 + args.n_perm)
    adj_topic = bh_fdr(p_topic)
    inter = pd.DataFrame({"Topic": K, "topic": [label(t) for t in K], "statistic": obs_topic,
                          "p_perm": p_topic, "p_fdr": adj_topic})
    tests = pd.DataFrame(rows_tests)
    tests["p_fdr"] = np.nan
    m = tests.test == "PageRank change, pairwise"
    tests.loc[m, "p_fdr"] = bh_fdr(tests.loc[m, "p_perm"].to_numpy())
    tag = args.tag
    pd.DataFrame(rows_pr).to_csv(OUT / f"prepost_pagerank{tag}.csv", index=False)
    pd.DataFrame(rows_topic).to_csv(OUT / f"prepost_pagerank_changes{tag}.csv", index=False)
    inter.to_csv(OUT / f"prepost_pagerank_interaction_by_topic{tag}.csv", index=False)
    tests.to_csv(OUT / f"prepost_pagerank_tests{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    pr = pd.DataFrame(rows_pr)
    print(pr.loc[pr.groupby(["condition", "time"]).pagerank.idxmax()][["condition", "time", "topic", "pagerank", "n_transitions"]].round(3).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    tc = pd.DataFrame(rows_topic)
    print(tc[(tc.p_perm < 0.05)].round(3).to_string(index=False))
    print(inter[inter.p_perm < 0.1].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
