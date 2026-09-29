#!/usr/bin/env python3
"""Between-condition tests of the pooled pre->post headline results.

For the measures that drive the pooled (all-conditions) pre->post effect, is the CHANGE different
between conditions?  Each participant contributes one paired change (post - pre):
  multivariate: topic composition (Hellinger vector), semantic space (centroid vector);
  univariate:   share of Anxiety & fear, Video & nature, Feeling calmer; PC1 and PC3 scores
                (the PCs that moved in the pooled test; PCs from all units, as in
                prepost_content_tests.py).
Contrasts: VR Art vs VR Only, VR Art vs Control, VR Only vs Control (BH-FDR over these three per
measure) and the planned combined contrast VR (VR Art + VR Only pooled) vs Control, reported
separately and BH-adjusted over the measures.
Tests: condition labels permuted across participants (10000). Effect = difference in mean change
(univariate, with a participant bootstrap 95% CI) or Euclidean distance between the mean change
vectors (multivariate, no CI: the bootstrap of a norm is biased upward). To compare measures on one axis the univariate changes are also
reported standardised by the SD of participants' change.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from prepost_common import CONDITIONS, OUT, TOPIC_LABELS, bh_fdr, embeddings, label, load_units, participants


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--drop-review", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    u = load_units(drop_review=args.drop_review)
    E = embeddings(u)
    P = participants(u)
    K = [t for t in sorted(TOPIC_LABELS) if t >= 0]
    kidx = {t: i for i, t in enumerate(K)}
    S = PCA(n_components=10, random_state=0).fit(E).transform(E)
    both = P[P.has_pre & P.has_post].reset_index(drop=True)
    lab = both.cond.to_numpy()

    def feats(pid, t):
        m = ((u.participant_id == pid) & (u.time == t)).to_numpy()
        cnt = np.bincount([kidx[x] for x in u.Topic[m]], minlength=len(K)).astype(float)
        c = E[m].mean(axis=0)
        return cnt / cnt.sum(), c / np.linalg.norm(c), S[m].mean(axis=0)

    f = {(i, t): feats(i, t) for i in both.participant_id for t in ("pre", "post")}
    pids = both.participant_id
    dshare = np.stack([f[(i, "post")][0] - f[(i, "pre")][0] for i in pids])
    dH = np.stack([np.sqrt(f[(i, "post")][0]) - np.sqrt(f[(i, "pre")][0]) for i in pids])
    dC = np.stack([f[(i, "post")][1] - f[(i, "pre")][1] for i in pids])
    dPC = np.stack([f[(i, "post")][2] - f[(i, "pre")][2] for i in pids])
    uni = {"Anxiety & fear (share)": dshare[:, kidx[0]], "Video & nature (share)": dshare[:, kidx[6]],
           "Feeling calmer (share)": dshare[:, kidx[8]], "PC1 (worry -> calm/fine)": dPC[:, 0],
           "PC3 (optimism/expecting well -> calm/video)": dPC[:, 2]}
    multi = {"Topic composition (Hellinger)": dH, "Semantic space (centroid)": dC}
    for k in uni:
        pass
    contrasts = [("VR Art vs VR Only", [0], [1]), ("VR Art vs Control", [0], [2]),
                 ("VR Only vs Control", [1], [2]), ("VR (Art + Only) vs Control", [0, 1], [2])]

    def diff_uni(x, l, a, b):
        return x[np.isin(l, a)].mean() - x[np.isin(l, b)].mean()

    def diff_multi(X, l, a, b):
        return np.linalg.norm(X[np.isin(l, a)].mean(axis=0) - X[np.isin(l, b)].mean(axis=0))

    rows = []
    for name, x in list(uni.items()) + list(multi.items()):
        is_multi = name in multi
        fn = diff_multi if is_multi else diff_uni
        sd = None if is_multi else x.std(ddof=1)
        for cname, a, b in contrasts:
            keep = np.isin(lab, a + b)
            idx = np.nonzero(keep)[0]
            obs = fn(x, lab, a, b)
            ge = 0
            for _ in range(args.n_perm):
                lp = lab.copy()
                lp[idx] = rng.permutation(lab[idx])
                v = fn(x, lp, a, b)
                ge += (v >= obs - 1e-12) if is_multi else (abs(v) >= abs(obs) - 1e-12)
            boots = []
            for _ in range(args.n_boot):
                s = np.concatenate([rng.choice(np.nonzero(np.isin(lab, g))[0], (np.isin(lab, g)).sum()) for g in (a, b)])
                # resample within the two sides of the contrast
                la = np.where(np.isin(lab[s], a), 0, 1)
                boots.append(fn(x[s], la, [0], [1]))
            # a norm of a difference is biased upward under resampling, so no CI for the multivariate effects
            lo, hi = (np.nan, np.nan) if is_multi else np.percentile(boots, [2.5, 97.5])
            rows.append({"measure": name, "type": "multivariate" if is_multi else "univariate", "contrast": cname,
                         "n_a": int(np.isin(lab, a).sum()), "n_b": int(np.isin(lab, b).sum()),
                         "mean_change_a": np.nan if is_multi else x[np.isin(lab, a)].mean(),
                         "mean_change_b": np.nan if is_multi else x[np.isin(lab, b)].mean(),
                         "difference": obs, "ci_low": lo, "ci_high": hi,
                         "std_difference": np.nan if is_multi else obs / sd,
                         "std_ci_low": np.nan if is_multi else lo / sd, "std_ci_high": np.nan if is_multi else hi / sd,
                         "p_perm": (1 + ge) / (1 + args.n_perm)})
    res = pd.DataFrame(rows)
    res["p_fdr_pairs"] = np.nan
    m = res.contrast != "VR (Art + Only) vs Control"
    res.loc[m, "p_fdr_pairs"] = res[m].groupby("measure")["p_perm"].transform(lambda s: bh_fdr(s.to_numpy()))
    m2 = ~m
    res["p_fdr_vr_vs_control"] = np.nan
    res.loc[m2, "p_fdr_vr_vs_control"] = bh_fdr(res.loc[m2, "p_perm"].to_numpy())
    res.to_csv(OUT / f"prepost_between_condition{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["measure", "contrast", "mean_change_a", "mean_change_b", "difference", "ci_low", "ci_high", "p_perm", "p_fdr_pairs", "p_fdr_vr_vs_control"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
