#!/usr/bin/env python3
"""Which principal components do the conditions differ on, and what do those PCs mean?

The PCs are the ones of the semantic-space map (PCA on all sentence embeddings).
Participants' scores = PC coordinates of their mean embedding.

* Group difference per PC (first ``--n-pc`` PCs, fixed in advance): permutation
  ANOVA F on participant scores (participant labels shuffled), eta^2 = share of
  variance between groups, Benjamini-Hochberg across the PCs. Pairwise
  (Holm over 3 pairs, permutation of mean differences) only for PCs whose
  omnibus test survives FDR < .05.
* What a PC means: score of each topic's centroid on the PC (in SD units of the
  sentence scores), plus the sentences at both ends. PC signs are arbitrary.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from all_questions.topic_labels import short_topic_label  # noqa: E402
from topic_network_analysis import CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, bh_fdr, load_data  # noqa: E402
from global_distribution_tests import holm  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-pc", type=int, default=10)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def anova_F(X, lab):
    """Column-wise one-way ANOVA F for scores X (n, k) and integer labels."""
    groups = np.unique(lab)
    grand = X.mean(axis=0)
    ssb = np.zeros(X.shape[1])
    ssw = np.zeros(X.shape[1])
    for g in groups:
        Xg = X[lab == g]
        mu = Xg.mean(axis=0)
        ssb += len(Xg) * (mu - grand) ** 2
        ssw += ((Xg - mu) ** 2).sum(axis=0)
    n, k = len(X), len(groups)
    return (ssb / (k - 1)) / (ssw / (n - k)), ssb / (ssb + ssw)


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    df = load_data(args.input, include_outlier=False)
    z = np.load(out / "sentence_embeddings.npz")
    lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
    E = np.stack([lookup[i] for i in df["document_id"]]).astype(np.float64)

    cond_of = df.drop_duplicates("participant_id").set_index("participant_id")["condition_label"]
    pids = list(cond_of.index)
    row_p = np.array([pids.index(x) for x in df["participant_id"]])
    cent = np.stack([E[row_p == i].mean(axis=0) for i in range(len(pids))])
    lab0 = np.array([CONDITIONS.index(cond_of[p]) for p in pids])

    K = args.n_pc
    pca = PCA(n_components=K, random_state=0).fit(E)
    S = pca.transform(E)
    P = pca.transform(cent)
    sd = S.std(axis=0)

    F_obs, eta_obs = anova_F(P, lab0)
    ge = np.zeros(K)
    for _ in range(args.n_perm):
        F_p, _ = anova_F(P, rng.permutation(lab0))
        ge += F_p >= F_obs - 1e-12
    p_perm = (1 + ge) / (1 + args.n_perm)
    p_fdr = bh_fdr(p_perm)

    rows = []
    boots = {c: np.stack([P[lab0 == c][rng.integers(0, (lab0 == c).sum(), (lab0 == c).sum())].mean(axis=0)
                          for _ in range(args.n_boot)]) for c in range(3)}
    for k in range(K):
        row = {"PC": k + 1, "variance_explained": pca.explained_variance_ratio_[k],
               "F": F_obs[k], "eta_squared": eta_obs[k], "p_perm": p_perm[k], "p_fdr": p_fdr[k]}
        for c, name in enumerate(CONDITIONS):
            m = P[lab0 == c, k].mean() / sd[k]
            lo, hi = np.percentile(boots[c][:, k], [2.5, 97.5]) / sd[k]
            row.update({f"mean_{name}": m, f"ci_low_{name}": lo, f"ci_high_{name}": hi})
        # pairwise only when the omnibus survives FDR
        pw = {}
        if p_fdr[k] < 0.05:
            ps = []
            for i, j in combinations(range(3), 2):
                both = np.nonzero(np.isin(lab0, (i, j)))[0]
                d_obs = abs(P[lab0 == i, k].mean() - P[lab0 == j, k].mean())
                cnt = 0
                for _ in range(args.n_perm):
                    lab = rng.permutation(lab0[both])
                    cnt += abs(P[both][lab == i, k].mean() - P[both][lab == j, k].mean()) >= d_obs - 1e-12
                ps.append((1 + cnt) / (1 + args.n_perm))
            for (i, j), ph in zip(combinations(range(3), 2), holm(np.array(ps))):
                pw[f"p_holm_{CONDITIONS[i]}_vs_{CONDITIONS[j]}"] = ph
        row.update(pw)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "pc_condition_tests.csv", index=False)

    topics = sorted(df["Topic"].unique())
    grand = E.mean(axis=0)
    load = np.array([[np.dot(E[(df["Topic"] == t).to_numpy()].mean(axis=0) - grand, pca.components_[k]) / sd[k]
                      for t in topics] for k in range(K)]).T  # (topics, K)
    ld = pd.DataFrame(load, index=[short_topic_label(t) for t in topics],
                      columns=[f"PC{k + 1}" for k in range(K)])
    ld.index.name = "topic"
    ld.to_csv(out / "pc_topic_loadings.csv")

    ex = []
    for k in range(K):
        order = np.argsort(S[:, k])
        for end, idx in (("low", order[:5]), ("high", order[::-1][:5])):
            for i in idx:
                ex.append({"PC": k + 1, "end": end, "score_sd": S[i, k] / sd[k],
                           "condition": df["condition_label"].iloc[i],
                           "topic": short_topic_label(df["Topic"].iloc[i]),
                           "sentence": df["sentence"].iloc[i]})
    pd.DataFrame(ex).to_csv(out / "pc_extreme_sentences.csv", index=False)

    show = res[["PC", "variance_explained", "F", "eta_squared", "p_perm", "p_fdr"]].round(4)
    print(show.to_string(index=False))
    print("cumulative variance of the first", K, "PCs:", round(float(pca.explained_variance_ratio_.sum()), 3))
    top = ld.apply(lambda c: pd.Series({"high": ", ".join(c.sort_values(ascending=False).index[:2]),
                                        "low": ", ".join(c.sort_values().index[:2])}))
    print(top.T.to_string())


if __name__ == "__main__":
    main()
