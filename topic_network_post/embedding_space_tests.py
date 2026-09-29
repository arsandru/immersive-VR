#!/usr/bin/env python3
"""Topic-free content tests in the sentence-embedding space (Qwen3 embeddings).

Do the conditions' sentences sit in different places in meaning-space?
Participants are the unit of permutation: labels are shuffled across
participants (group sizes kept), so sentences from one person stay together.
Omnibus = all three conditions; pairwise p-values use Holm over the 3 pairs.

  * PERMANOVA on sentence-level cosine distances (pseudo-F)
  * PERMANOVA on participant centroids (Euclid on unit vectors)
  * dispersion (distance of participant centroids to their group centroid)
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_network_analysis import CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, load_data  # noqa: E402
from global_distribution_tests import holm  # noqa: E402

PAIRS = list(combinations(range(3), 2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def anova_F(values, lab, groups):
    idx = [np.nonzero(lab == g)[0] for g in groups]
    allv = np.concatenate([values[i] for i in idx])
    grand = allv.mean()
    ssb = sum(len(i) * (values[i].mean() - grand) ** 2 for i in idx)
    ssw = sum(((values[i] - values[i].mean()) ** 2).sum() for i in idx)
    n, k = len(allv), len(groups)
    return (ssb / (k - 1)) / (ssw / (n - k)) if ssw > 0 else 0.0


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    df = load_data(args.input, include_outlier=False)
    z = np.load(out / "sentence_embeddings.npz")
    lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
    E = np.stack([lookup[i] for i in df["document_id"]]).astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    cond_of = df.drop_duplicates("participant_id").set_index("participant_id")["condition_label"]
    pids = [p for c in CONDITIONS for p in cond_of.index[cond_of == c]]
    N = len(pids)
    lab0 = np.array([CONDITIONS.index(cond_of[p]) for p in pids])
    pidx = {p: i for i, p in enumerate(pids)}
    row_p = np.array([pidx[p] for p in df["participant_id"]])

    dist = 1 - np.einsum("id,jd->ij", E, E)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    cent = np.stack([E[row_p == i].mean(axis=0) for i in range(N)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)

    # statistics: f(lab, groups) with lab in {-1, 0, 1, 2} per participant
    def s_permanova_sentence(lab, groups):
        sl = lab[row_p]
        sg = [np.nonzero(sl == g)[0] for g in groups]
        allx = np.concatenate(sg)
        n = len(allx)
        d2 = dist[np.ix_(allx, allx)] ** 2
        sst = d2[np.triu_indices(n, 1)].sum() / n
        ssw = sum((dist[np.ix_(i, i)][np.triu_indices(len(i), 1)] ** 2).sum() / len(i) for i in sg)
        return ((sst - ssw) / (len(groups) - 1)) / (ssw / (n - len(groups)))

    def s_permanova_centroid(lab, groups):
        idx = [np.nonzero(lab == g)[0] for g in groups]
        X = np.vstack([cent[i] for i in idx])
        grand = X.mean(axis=0)
        ssb = sum(len(i) * ((cent[i].mean(axis=0) - grand) ** 2).sum() for i in idx)
        ssw = sum(((cent[i] - cent[i].mean(axis=0)) ** 2).sum() for i in idx)
        n = len(X)
        return (ssb / (len(groups) - 1)) / (ssw / (n - len(groups)))

    def s_dispersion(lab, groups):
        d = np.full(N, np.nan)
        for g in groups:
            i = np.nonzero(lab == g)[0]
            d[i] = np.linalg.norm(cent[i] - cent[i].mean(axis=0), axis=1)
        return anova_F(d, lab, groups)

    tests = [("sentence PERMANOVA (cosine), pseudo-F", s_permanova_sentence),
             ("participant-centroid PERMANOVA, pseudo-F", s_permanova_centroid),
             ("dispersion of participant centroids, F", s_dispersion)]
    comparisons = [("all three", (0, 1, 2))] + [(f"{CONDITIONS[i]} vs {CONDITIONS[j]}", (i, j)) for i, j in PAIRS]
    rows = []
    for name, fn in tests:
        for cname, groups in comparisons:
            keep = np.isin(lab0, groups)
            lab = np.where(keep, lab0, -1)
            obs = fn(lab, groups)
            members = np.nonzero(keep)[0]
            ge = 0
            for _ in range(args.n_perm):
                lab_p = np.full(N, -1)
                lab_p[members] = rng.permutation(lab0[members])
                ge += fn(lab_p, groups) >= obs - 1e-12
            rows.append({"test": name, "comparison": cname, "statistic": obs,
                         "p_perm": (1 + ge) / (1 + args.n_perm)})
        print(f"done: {name}", flush=True)
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    pair_mask = res.comparison != "all three"
    res.loc[pair_mask, "p_holm"] = res[pair_mask].groupby("test")["p_perm"].transform(
        lambda s: holm(s.to_numpy()))
    res.to_csv(out / "embedding_space_tests.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
