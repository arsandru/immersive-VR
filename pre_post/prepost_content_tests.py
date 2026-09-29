#!/usr/bin/env python3
"""Pre vs post content tests: topic composition and semantic-space shift.

Units = Q1 descriptors (see prepare_and_model.py); the topic model is shared by pre and post.
Participants are the unit of analysis; paired tests use the participants observed at both
timepoints (n = 49).

BASELINE (pre only, all pre participants): do the conditions differ before the intervention?
  * participant-level PERMANOVA on Hellinger topic proportions, and on participant embedding
    centroids; pairwise BH-FDR.
TIME (paired, per condition and pooled): does content move from pre to post?
  * composition: statistic = squared norm of the mean paired difference of Hellinger topic
    vectors; embedding: same on participant centroids. Null: random sign flips of each
    participant's difference (equivalent to swapping pre/post labels within participant).
  * per topic: mean change in topic share, sign-flip p, BH within condition.
  * per PC (first 10 PCs of all units, fixed in advance): mean change, sign-flip p, BH.
INTERACTION (does the change differ by condition?): PERMANOVA pseudo-F on the paired
  difference vectors, condition labels permuted across participants; pairwise BH-FDR;
  per-topic and per-PC one-way ANOVA F on the change, BH.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from prepost_common import (CONDITIONS, OUT, TOPIC_LABELS, bh_fdr, embeddings, label,
                            load_units, participants)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-pc", type=int, default=10)
    p.add_argument("--drop-review", action="store_true", help="sensitivity: drop needs_review rows")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pseudo_F(X, lab, groups):
    """PERMANOVA pseudo-F with Euclidean distance on rows of X."""
    grand = X[np.isin(lab, groups)].mean(axis=0)
    ssb = ssw = 0.0
    n = 0
    for g in groups:
        Xg = X[lab == g]
        mu = Xg.mean(axis=0)
        ssb += len(Xg) * ((mu - grand) ** 2).sum()
        ssw += ((Xg - mu) ** 2).sum()
        n += len(Xg)
    return (ssb / (len(groups) - 1)) / (ssw / (n - len(groups)))


def anova_cols(X, lab, groups):
    grand = X[np.isin(lab, groups)].mean(axis=0)
    ssb = np.zeros(X.shape[1])
    ssw = np.zeros(X.shape[1])
    n = 0
    for g in groups:
        Xg = X[lab == g]
        mu = Xg.mean(axis=0)
        ssb += len(Xg) * (mu - grand) ** 2
        ssw += ((Xg - mu) ** 2).sum(axis=0)
        n += len(Xg)
    return (ssb / (len(groups) - 1)) / np.maximum(ssw / (n - len(groups)), 1e-12)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    u = load_units(drop_review=args.drop_review)
    E = embeddings(u)
    P = participants(u)
    K = sorted(TOPIC_LABELS)
    K = [t for t in K if t >= 0]
    kidx = {t: i for i, t in enumerate(K)}
    pca = PCA(n_components=args.n_pc, random_state=0).fit(E)
    Spc = pca.transform(E)

    def features(pid, time):
        m = ((u.participant_id == pid) & (u.time == time)).to_numpy()
        if not m.any():
            return None
        cnt = np.bincount([kidx[t] for t in u.Topic[m]], minlength=len(K)).astype(float)
        c = E[m].mean(axis=0)
        return cnt / cnt.sum(), c / np.linalg.norm(c), Spc[m].mean(axis=0)

    feat = {(r.participant_id, t): features(r.participant_id, t) for r in P.itertuples() for t in ("pre", "post")}
    out_rows = []

    def add(**kw):
        out_rows.append(kw)

    # ---------------- baseline (pre only)
    pre = P[P.has_pre].reset_index(drop=True)
    H = np.stack([np.sqrt(feat[(i, "pre")][0]) for i in pre.participant_id])
    C = np.stack([feat[(i, "pre")][1] for i in pre.participant_id])
    lab = pre.cond.to_numpy()
    for name, X in (("topic composition (Hellinger PERMANOVA)", H), ("semantic space (centroid PERMANOVA)", C)):
        for cname, groups in [("all three", (0, 1, 2))] + [(f"{CONDITIONS[i]} vs {CONDITIONS[j]}", (i, j)) for i, j in combinations(range(3), 2)]:
            keep = np.isin(lab, groups)
            obs = pseudo_F(X, lab, groups)
            members = np.nonzero(keep)[0]
            ge = 0
            for _ in range(args.n_perm):
                lp = lab.copy()
                lp[members] = rng.permutation(lab[members])
                ge += pseudo_F(X, lp, groups) >= obs - 1e-12
            add(analysis="baseline (pre only)", test=name, comparison=cname, n=int(keep.sum()),
                statistic=obs, p_perm=(1 + ge) / (1 + args.n_perm))

    # ---------------- paired
    both = P[P.has_pre & P.has_post].reset_index(drop=True)
    labp = both.cond.to_numpy()
    dH = np.stack([np.sqrt(feat[(i, "post")][0]) - np.sqrt(feat[(i, "pre")][0]) for i in both.participant_id])
    dS = np.stack([feat[(i, "post")][2] - feat[(i, "pre")][2] for i in both.participant_id])  # PC scores
    dC = np.stack([feat[(i, "post")][1] - feat[(i, "pre")][1] for i in both.participant_id])
    dP = np.stack([feat[(i, "post")][0] - feat[(i, "pre")][0] for i in both.participant_id])  # share change

    def signflip(D, p_cols=False):
        obs = (D.mean(axis=0) ** 2).sum() if not p_cols else D.mean(axis=0)
        cnt = 0 if not p_cols else np.zeros(D.shape[1])
        for _ in range(args.n_perm):
            s = rng.choice([-1.0, 1.0], size=(len(D), 1))
            v = (s * D).mean(axis=0)
            cnt += ((v ** 2).sum() >= obs - 1e-12) if not p_cols else (np.abs(v) >= np.abs(obs) - 1e-12)
        return obs, (1 + cnt) / (1 + args.n_perm)

    groups_time = [("all conditions", np.ones(len(both), bool))] + [(c, labp == i) for i, c in enumerate(CONDITIONS)]
    topic_rows, pc_rows = [], []
    for cname, m in groups_time:
        for name, D in (("topic composition (Hellinger)", dH[m]), ("semantic space (centroid)", dC[m])):
            obs, p = signflip(D)
            add(analysis="pre vs post (paired)", test=name, comparison=cname, n=int(m.sum()),
                statistic=obs, p_perm=p)
        mean, p = signflip(dP[m], p_cols=True)
        adj = bh_fdr(p)
        for j, t in enumerate(K):
            topic_rows.append({"condition": cname, "Topic": t, "topic": label(t),
                               "share_pre": np.mean([feat[(i, "pre")][0][j] for i in both.participant_id[m]]),
                               "share_post": np.mean([feat[(i, "post")][0][j] for i in both.participant_id[m]]),
                               "change": mean[j], "p_perm": p[j], "p_fdr": adj[j], "n": int(m.sum())})
        mean, p = signflip(dS[m], p_cols=True)
        adj = bh_fdr(p)
        for k in range(args.n_pc):
            pc_rows.append({"condition": cname, "PC": k + 1, "change_sd": mean[k] / Spc[:, k].std(),
                            "p_perm": p[k], "p_fdr": adj[k], "n": int(m.sum())})
    # interaction
    for name, D in (("topic composition (Hellinger)", dH), ("semantic space (centroid)", dC)):
        for cname, groups in [("all three", (0, 1, 2))] + [(f"{CONDITIONS[i]} vs {CONDITIONS[j]}", (i, j)) for i, j in combinations(range(3), 2)]:
            keep = np.isin(labp, groups)
            obs = pseudo_F(D, labp, groups)
            members = np.nonzero(keep)[0]
            ge = 0
            for _ in range(args.n_perm):
                lp = labp.copy()
                lp[members] = rng.permutation(labp[members])
                ge += pseudo_F(D, lp, groups) >= obs - 1e-12
            add(analysis="interaction (change differs by condition)", test=name, comparison=cname,
                n=int(keep.sum()), statistic=obs, p_perm=(1 + ge) / (1 + args.n_perm))
    for name, D, rows_ in (("topic", dP, topic_rows), ("PC", dS, pc_rows)):
        obs = anova_cols(D, labp, (0, 1, 2))
        cnt = np.zeros(D.shape[1])
        for _ in range(args.n_perm):
            cnt += anova_cols(D, rng.permutation(labp), (0, 1, 2)) >= obs - 1e-12
        p = (1 + cnt) / (1 + args.n_perm)
        adj = bh_fdr(p)
        for j in range(D.shape[1]):
            rows_.append({"condition": "interaction (ANOVA F)", "Topic" if name == "topic" else "PC": K[j] if name == "topic" else j + 1,
                          **({"topic": label(K[j])} if name == "topic" else {}),
                          "F": obs[j], "p_perm": p[j], "p_fdr": adj[j], "n": len(both)})
    pairs = pd.DataFrame(out_rows)
    pairs["p_fdr"] = np.nan
    mask = (pairs.comparison != "all three") & pairs.analysis.isin(["baseline (pre only)", "interaction (change differs by condition)"])
    pairs.loc[mask, "p_fdr"] = pairs[mask].groupby(["analysis", "test"])["p_perm"].transform(lambda s: bh_fdr(s.to_numpy()))
    tag = args.tag
    pairs.to_csv(OUT / f"prepost_content_tests{tag}.csv", index=False)
    pd.DataFrame(topic_rows).to_csv(OUT / f"prepost_topic_changes{tag}.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(OUT / f"prepost_pc_changes{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(pairs.round(4).to_string(index=False))
    tr = pd.DataFrame(topic_rows)
    print(tr[(tr.p_fdr < 0.1) | (tr.p_perm < 0.01)].round(3).to_string(index=False))
    pc = pd.DataFrame(pc_rows)
    print(pc[pc.p_fdr < 0.1].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
