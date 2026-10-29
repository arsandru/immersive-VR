#!/usr/bin/env python3
"""Nearest-neighbour condition-mixing test in the sentence-embedding space (exploratory; no dimension reduction).

Question: do a sentence's nearest neighbours in meaning-space come from the same condition more often than chance?

Statistic (fixed in advance): for every sentence, take its k = 10 nearest sentences by cosine similarity in the full Qwen3
embedding, EXCLUDING sentences of the same participant (so a person's own sentences cannot make the conditions look separate);
the score of the sentence is the share of those neighbours that carry its own condition label; scores are averaged within
each participant and then across participants (every participant weighs the same). Omnibus = all three conditions together;
pairwise = only the two conditions' sentences, neighbours drawn from that pair.

Null: participant labels are shuffled across participants (group sizes kept), neighbours stay fixed, so sentences of one
participant stay together (same scheme as embedding_space_tests.py). One-sided (more same-condition neighbours than chance);
pairs: Holm over the three. Excess = observed minus the permutation mean (chance level depends on group sizes).

Sensitivity: k = 5 and 20; sentence-weighted average instead of participant-weighted.
Per-condition rows (omnibus neighbours, share of same-condition neighbours for that condition's sentences) are descriptive;
BH over the three conditions.

Outputs: knn_mixing_tests.csv, knn_mixing_tests.[pdf|png]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from topic_network_analysis import CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, load_data  # noqa: E402
from global_distribution_tests import holm  # noqa: E402

K_MAIN = 10
K_SENS = (5, 20)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1)
    return out


def neighbours(sim, rows, pid, k):
    """k nearest sentences (among `rows`, other participants only) for every sentence in `rows`; indices into `rows`."""
    s = sim[np.ix_(rows, rows)].copy()
    s[pid[rows][:, None] == pid[rows][None, :]] = -np.inf
    return np.argsort(-s, axis=1)[:, :k]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)

    df = load_data(args.input, include_outlier=False)
    z = np.load(out / "sentence_embeddings.npz")
    lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
    E = np.stack([lookup[i] for i in df["document_id"]]).astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    sim = np.einsum("id,jd->ij", E, E)  # einsum: E @ E.T raises spurious overflow warnings with this numpy build

    cond_of = df.drop_duplicates("participant_id").set_index("participant_id")["condition_label"]
    pids = [q for c in CONDITIONS for q in cond_of.index[cond_of == c]]
    N = len(pids)
    lab0 = np.array([CONDITIONS.index(cond_of[q]) for q in pids])
    pidx = {q: i for i, q in enumerate(pids)}
    row_p = np.array([pidx[q] for q in df["participant_id"]])
    print(f"{len(df)} sentences, {N} participants; per condition:",
          {CONDITIONS[g]: int((lab0 == g).sum()) for g in range(3)}, flush=True)

    def score(lab, rows, nbr, weighted):
        """share of same-condition neighbours, averaged over participants (or sentences); lab = participant labels."""
        L = lab[row_p[rows]]
        same = (L[nbr] == L[:, None]).mean(axis=1)
        if weighted == "sentence":
            return same.mean(), same
        pr = row_p[rows]
        cnt = np.bincount(pr, minlength=N)
        pm = np.bincount(pr, weights=same, minlength=N)
        have = cnt > 0
        return (pm[have] / cnt[have]).mean(), same

    comparisons = [("all three", (0, 1, 2))] + [(f"{CONDITIONS[i]} vs {CONDITIONS[j]}", (i, j)) for i, j in combinations(range(3), 2)]
    rows_out = []
    null_store = {}
    variants = [("main", K_MAIN, "participant")] + [(f"k = {k}", k, "participant") for k in K_SENS] + [("sentence-weighted", K_MAIN, "sentence")]
    for vname, k, weighted in variants:
        for cname, groups in comparisons:
            members = np.nonzero(np.isin(lab0, groups))[0]
            srows = np.nonzero(np.isin(lab0[row_p], groups))[0]
            nbr = neighbours(sim, srows, row_p, k)
            lab_full = lab0.copy()
            obs, same_obs = score(lab_full, srows, nbr, weighted)
            null = np.empty(args.n_perm)
            cond_null = np.empty((args.n_perm, 3))
            for b in range(args.n_perm):
                lab_p = np.full(N, -1)
                lab_p[members] = rng.permutation(lab0[members])
                null[b], same_b = score(lab_p, srows, nbr, weighted)
                if cname == "all three" and vname == "main":
                    Lb = lab_p[row_p[srows]]
                    cond_null[b] = [same_b[Lb == g].mean() for g in range(3)]
            pval = (1 + (null >= obs - 1e-12).sum()) / (1 + args.n_perm)
            rows_out.append(dict(variant=vname, k=k, comparison=cname, level="overall", observed=obs, null_mean=null.mean(),
                                 null_sd=null.std(), excess=obs - null.mean(), p_perm=pval, n_sentences=len(srows),
                                 n_participants=len(members)))
            if vname == "main":
                null_store[cname] = (obs, null, pval)
            if cname == "all three" and vname == "main":
                L0 = lab0[row_p[srows]]
                pc = []
                for g in range(3):
                    o = same_obs[L0 == g].mean()
                    pg = (1 + (cond_null[:, g] >= o - 1e-12).sum()) / (1 + args.n_perm)
                    pc.append(pg)
                    rows_out.append(dict(variant="main", k=k, comparison=f"{CONDITIONS[g]} sentences", level="per condition (descriptive)",
                                         observed=o, null_mean=cond_null[:, g].mean(), null_sd=cond_null[:, g].std(),
                                         excess=o - cond_null[:, g].mean(), p_perm=pg, n_sentences=int((L0 == g).sum()),
                                         n_participants=int((lab0 == g).sum())))
                for r, pa in zip(rows_out[-3:], bh(pc)):
                    r["p_bh"] = pa
        print(f"done: {vname}", flush=True)

    res = pd.DataFrame(rows_out)
    res["p_holm"] = np.nan
    for v in res.variant.unique():
        m = (res.variant == v) & res.comparison.str.contains(" vs ") & (res.level == "overall")
        res.loc[m, "p_holm"] = holm(res.loc[m, "p_perm"].to_numpy())
    res.to_csv(out / "knn_mixing_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(res[["variant", "comparison", "level", "observed", "null_mean", "excess", "p_perm", "p_holm", "p_bh"]].round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
    for ax, (cname, _) in zip(axes, comparisons):
        obs, null, pval = null_store[cname]
        ax.hist(null, bins=40, color="#bbbbbb", edgecolor="white")
        ax.axvline(obs, color="#c0392b", lw=2)
        ph = res[(res.variant == "main") & (res.comparison == cname) & (res.level == "overall")].p_holm.iloc[0]
        ax.set_title(cname + f"\nobs {obs:.3f}, chance {null.mean():.3f}, p = {pval:.3f}" + ("" if np.isnan(ph) else f", Holm {ph:.3f}"), fontsize=9.5)
        ax.set_yticks([])
        ax.set_xlabel("share of same-condition neighbours")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Nearest-neighbour condition mixing (k = {K_MAIN}, other participants only; grey = participant-label permutation null; exploratory)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "knn_mixing_tests.pdf", bbox_inches="tight")
    fig.savefig(out / "knn_mixing_tests.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
