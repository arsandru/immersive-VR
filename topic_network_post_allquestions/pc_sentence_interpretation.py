#!/usr/bin/env python3
"""Sentence-based interpretation of the condition-separating PCs (PC4, PC6); no topic model involved (exploratory).

PCs = PCA of the Qwen3 sentence embeddings (10 components, as in pc_differences.py). Settings fixed before running.

  A  Representative sentences per PC end. Ends = top / bottom quartile of the sentence scores. Within an end, sentences are
     ranked by cosine similarity to the end's mean embedding (how typical they are of that end); at most one sentence per
     participant; the 8 most typical are kept.
  B  Word association on ALL sentences: for each word used in sentences of >= 5 participants, the difference in mean PC score
     (SD units) between sentences that contain the word and sentences that do not, with a participant-cluster bootstrap 95% CI
     (2000 resamples) and a bootstrap p-value, Benjamini-Hochberg over the words of each PC. Words are lower-cased alphabetic
     tokens of >= 3 letters; function words are removed (the standard Portuguese stop-word list of q3/, without its evaluative
     terms such as "bem" or "bom", which matter here); no stemming, so inflected forms are separate words.
  C  Nuisance checks: association of each PC with sentence length (Spearman, participant-cluster bootstrap CI) and with question
     (eta^2), and the condition difference on the PC within each question (participant-level permutation ANOVA, 10000
     permutations, BH over the 6 tests; Q2 has only 6 to 10 participants per group).

Outputs: pc_sentence_representatives.csv, pc_sentence_words.csv, pc_sentence_nuisance.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from q3.q3_topic_analysis import PORTUGUESE_STOP_WORDS as _SW  # noqa: E402
from topic_network_analysis import CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, bh_fdr, load_data  # noqa: E402
from pc_differences import anova_F  # noqa: E402

STOP = set(_SW[: _SW.index("algum")])  # function words only (the rest of the list are evaluative/discourse terms)
PCS = (4, 6)
N_REP, MIN_PART, N_BOOT, N_PERM = 8, 5, 2000, 10000


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-záàâãéêíóôõúç]+", str(text).lower()) if len(w) >= 3 and w not in STOP}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)

    df = load_data(args.input, include_outlier=False)
    z = np.load(out / "sentence_embeddings.npz")
    lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
    E = np.stack([lookup[i] for i in df["document_id"]]).astype(np.float64)
    pca = PCA(n_components=10, random_state=0).fit(E)
    S = pca.transform(E)
    S = S / S.std(axis=0)  # SD units of each PC
    cond_of = df.drop_duplicates("participant_id").set_index("participant_id")["condition_label"]
    pids = list(cond_of.index)
    lab0 = np.array([CONDITIONS.index(cond_of[p]) for p in pids])
    row_p = np.array([pids.index(x) for x in df["participant_id"]])
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    n_words_sent = df["sentence"].astype(str).str.split().str.len().to_numpy()
    toks = [tokens(t) for t in df["sentence"]]

    # ---- A: representative sentences
    rep = []
    for pc in PCS:
        s = S[:, pc - 1]
        lo, hi = np.quantile(s, [0.25, 0.75])
        for end, m in (("high", s >= hi), ("low", s <= lo)):
            idx = np.nonzero(m)[0]
            c = En[idx].mean(axis=0)
            c /= np.linalg.norm(c)
            order = idx[np.argsort(-(En[idx] @ c))]
            seen, rank = set(), 0
            for i in order:
                if row_p[i] in seen:
                    continue
                seen.add(row_p[i])
                rank += 1
                rep.append(dict(PC=pc, end=end, rank=rank, typicality_cosine=float(En[i] @ c), score_sd=s[i],
                                participant_id=df.participant_id.iloc[i], condition=df.condition_label.iloc[i],
                                question=df.question_id.iloc[i], n_words=int(n_words_sent[i]), sentence=df.sentence.iloc[i]))
                if rank == N_REP:
                    break
    pd.DataFrame(rep).to_csv(out / "pc_sentence_representatives.csv", index=False)

    # ---- B: word association (all sentences), participant-cluster bootstrap
    part_n = {}
    for i, t in enumerate(toks):
        for w in t:
            part_n.setdefault(w, set()).add(row_p[i])
    vocab = sorted(w for w, ps in part_n.items() if len(ps) >= MIN_PART)
    X = np.zeros((len(df), len(vocab)))
    wi = {w: j for j, w in enumerate(vocab)}
    for i, t in enumerate(toks):
        for w in t:
            if w in wi:
                X[i, wi[w]] = 1
    members = [np.nonzero(row_p == p)[0] for p in range(len(pids))]
    P = len(pids)

    def diffs(s, wt):
        n = wt.sum()
        c = X.T @ wt
        has = X.T @ (wt * s)
        tot = (wt * s).sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            return has / c - (tot - has) / (n - c)

    word_rows = []
    boot_w = np.stack([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(N_BOOT)])
    for pc in PCS:
        s = S[:, pc - 1]
        obs = diffs(s, np.ones(len(df)))
        B = np.stack([diffs(s, bw[row_p].astype(float)) for bw in boot_w])
        lo, hi = np.nanpercentile(B, [2.5, 97.5], axis=0)
        p_boot = np.minimum(1, 2 * np.minimum((np.nan_to_num(B, nan=0) <= 0).mean(axis=0), (np.nan_to_num(B, nan=0) >= 0).mean(axis=0)))
        p_boot = np.maximum(p_boot, 1 / (N_BOOT + 1))
        pf = bh_fdr(p_boot)
        for j, w in enumerate(vocab):
            word_rows.append(dict(PC=pc, word=w, n_sentences=int(X[:, j].sum()), n_participants=len(part_n[w]), mean_diff_sd=obs[j],
                                  ci_low=lo[j], ci_high=hi[j], p_boot=p_boot[j], p_fdr=pf[j], direction="high" if obs[j] > 0 else "low"))
    words = pd.DataFrame(word_rows)
    words.to_csv(out / "pc_sentence_words.csv", index=False)

    # ---- C: nuisance checks
    rows = []
    qs = sorted(df.question_id.unique())
    for pc in PCS:
        s = S[:, pc - 1]
        rho = spearmanr(n_words_sent, s)[0]
        bt = []
        for bw in boot_w[:1000]:
            idx = np.repeat(np.arange(len(df)), bw[row_p])
            bt.append(spearmanr(n_words_sent[idx], s[idx])[0])
        rows.append(dict(check="sentence length (words), Spearman rho", PC=pc, question="all", statistic=rho,
                         ci_low=np.percentile(bt, 2.5), ci_high=np.percentile(bt, 97.5)))
        grand = s.mean()
        ssb = sum((df.question_id == q).sum() * (s[(df.question_id == q).to_numpy()].mean() - grand) ** 2 for q in qs)
        rows.append(dict(check="question, eta^2 (descriptive)", PC=pc, question="Q1-Q3", statistic=ssb / ((s - grand) ** 2).sum()))
        for q in qs:
            m = (df.question_id == q).to_numpy()
            pp = np.unique(row_p[m])
            Pq = np.array([[s[m & (row_p == p)].mean()] for p in pp])
            lab = lab0[pp]
            F, _ = anova_F(Pq, lab)
            ge = sum(anova_F(Pq, rng.permutation(lab))[0][0] >= F[0] - 1e-12 for _ in range(N_PERM))
            means = [Pq[lab == g, 0].mean() for g in range(3)]
            rows.append(dict(check="condition difference within question (permutation ANOVA)", PC=pc, question=q, statistic=F[0],
                             p_perm=(1 + ge) / (1 + N_PERM), n_VR_Art=int((lab == 0).sum()), n_VR_Only=int((lab == 1).sum()),
                             n_Control=int((lab == 2).sum()), mean_VR_Art=means[0], mean_VR_Only=means[1], mean_Control=means[2]))
    nu = pd.DataFrame(rows)
    m = nu.check.str.startswith("condition difference")
    nu.loc[m, "p_bh"] = bh_fdr(nu.loc[m, "p_perm"].to_numpy())
    nu.to_csv(out / "pc_sentence_nuisance.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"{len(df)} sentences, vocabulary {len(vocab)} words (>= {MIN_PART} participants)")
    for pc in PCS:
        w = words[words.PC == pc]
        sig = w[(w.ci_low > 0) | (w.ci_high < 0)].sort_values("mean_diff_sd")
        print(f"\nPC{pc}: words with bootstrap CI excluding 0 ({len(sig)}); FDR<.05: {int((w.p_fdr < .05).sum())}")
        print("  low :", ", ".join(f"{r.word}({r.mean_diff_sd:+.2f},n={r.n_participants})" for r in sig[sig.direction == 'low'].head(12).itertuples()))
        print("  high:", ", ".join(f"{r.word}({r.mean_diff_sd:+.2f},n={r.n_participants})" for r in sig[sig.direction == 'high'].tail(12)[::-1].itertuples()))
    print()
    print(nu.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
