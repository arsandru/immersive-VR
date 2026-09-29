#!/usr/bin/env python3
"""Evidence for what sits at each end of the PCs that separate the conditions.

Three independent, pre-specified checks (PCs = those with omnibus FDR < .05 in
pc_condition_tests.csv; ends = top / bottom quartile of sentence scores):

1. Distinctive words: weighted log-odds ratio with an informative Dirichlet prior
   (Monroe, Colaresi & Quinn 2008), z-scored, top-quartile vs bottom-quartile
   sentences. A word counts as *supported* when |z| > 1.96, it occurs in >= 3
   participants, and its sign and |z| > 1.96 hold in >= 90% of participant-level
   cluster bootstraps (2000).
2. Topic positions: mean PC score of each topic's sentences (in SD units) with
   a participant-cluster bootstrap 95% CI; a topic sits at an end when its CI
   excludes 0.
3. Who is at each end: share of end sentences from each condition versus the
   condition's share of all sentences.

Labels (PC_END_LABELS below) are written after inspecting this evidence by a fixed
rule: the label must be consistent with the topics whose bootstrap CI excludes 0 at
that end (largest |position| first) and must not contradict the leading words. With
~80 sentences per end no single word reaches the strict support threshold, so the
words only corroborate the direction; the topic positions carry the evidence.
The labels are an interpretation and are stored with their evidence in
pc_end_labels.csv.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from all_questions.topic_labels import short_topic_label  # noqa: E402
from topic_network_analysis import CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, load_data  # noqa: E402


PC_END_LABELS = {
    (4, "high"): "Enjoyment",
    (4, "low"): "Worries (surgery, care)",
    (6, "high"): "Relief",
    (6, "low"): "Worries (anxiety)",
}
# keyword check written after seeing the evidence (exploratory): share of sentences per end
# that contain any stem of a lexicon; stems are matched inside the cleaned Portuguese tokens
LEXICONS = {
    "enjoyment": r"gost|bonit|adoro|agrad|interess|lind|divert|boa\b|boas\b|bom\b|bons\b|maravilh",
    "relief": r"calm|tranquil|relax|descontra|alivi|confiante|sossego|paz",
    "worry": r"ansi|medo|preocup|nervos|receio|angústia|stress|cirurgia|operad|espera",
}
GLOSS = {"vídeo": "video", "gostei": "liked", "natureza": "nature", "ar": "air", "água": "water",
         "imagens": "images", "filme": "film", "vamos": "let's go", "sítio": "place", "sensação": "sensation",
         "cirurgia": "surgery", "ansiedade": "anxiety", "preocupada": "worried", "preocupação": "worry",
         "cuidado": "care", "espera": "waiting", "confiante": "confident", "fazer": "do", "vai": "goes",
         "calma": "calm", "relaxada": "relaxed", "sinto": "I feel", "ansiosa": "anxious", "medo": "fear",
         "pode": "can", "ainda": "still", "pessoas": "people"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--alpha0", type=float, default=200.0, help="total prior pseudo-count")
    p.add_argument("--min-count", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def log_odds_z(ya, yb, prior):
    na, nb, a0 = ya.sum(), yb.sum(), prior.sum()
    la = np.log((ya + prior) / (na + a0 - ya - prior))
    lb = np.log((yb + prior) / (nb + a0 - yb - prior))
    return (la - lb) / np.sqrt(1 / (ya + prior) + 1 / (yb + prior))


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    df = load_data(args.input, include_outlier=False).reset_index(drop=True)
    z = np.load(out / "sentence_embeddings.npz")
    lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
    E = np.stack([lookup[i] for i in df["document_id"]]).astype(np.float64)
    tests = pd.read_csv(out / "pc_condition_tests.csv")
    pcs = tests.loc[tests.p_fdr < 0.05, "PC"].astype(int).tolist()
    pca = PCA(n_components=int(max(pcs)), random_state=0).fit(E)
    S = pca.transform(E)
    sd = S.std(axis=0)

    # bag of words from the cleaned content tokens
    toks = df["content_text"].fillna("").str.split()
    vocab_counts = pd.Series([w for t in toks for w in set(t)]).value_counts()
    vocab = vocab_counts[vocab_counts >= args.min_count].index.tolist()
    widx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(df), len(vocab)))
    for r, t in enumerate(toks):
        for w in t:
            if w in widx:
                X[r, widx[w]] += 1
    pid = df["participant_id"].to_numpy()
    upid = np.unique(pid)
    rows_of = {p: np.nonzero(pid == p)[0] for p in upid}
    prior = args.alpha0 * X.sum(axis=0) / X.sum()

    word_rows, topic_rows, who_rows = [], [], []
    for k in pcs:
        s = S[:, k - 1]
        lo_cut, hi_cut = np.percentile(s, [25, 75])
        hi = s >= hi_cut
        lo = s <= lo_cut
        ya, yb = X[hi].sum(axis=0), X[lo].sum(axis=0)
        zobs = log_odds_z(ya, yb, prior)
        # participant-cluster bootstrap of the z scores (end sets fixed)
        stable = np.zeros(len(vocab))
        for _ in range(args.n_boot):
            samp = np.concatenate([rows_of[p] for p in rng.choice(upid, len(upid))])
            h, l = samp[hi[samp]], samp[lo[samp]]
            zb = log_odds_z(X[h].sum(axis=0), X[l].sum(axis=0), prior)
            stable += (np.sign(zb) == np.sign(zobs)) & (np.abs(zb) > 1.96)
        stable /= args.n_boot
        n_part_hi = np.array([len(set(pid[hi & (X[:, j] > 0)])) for j in range(len(vocab))])
        n_part_lo = np.array([len(set(pid[lo & (X[:, j] > 0)])) for j in range(len(vocab))])
        for j, w in enumerate(vocab):
            end = "high" if zobs[j] > 0 else "low"
            npart = n_part_hi[j] if end == "high" else n_part_lo[j]
            word_rows.append({"PC": k, "end": end, "word": w, "z": zobs[j],
                              "count_high": ya[j], "count_low": yb[j], "n_participants": npart,
                              "bootstrap_stability": stable[j],
                              "supported": bool(abs(zobs[j]) > 1.96 and npart >= 3 and stable[j] >= 0.90)})
        # topic positions with cluster bootstrap CI
        topics = sorted(df["Topic"].unique())
        for t in topics:
            m = (df["Topic"] == t).to_numpy()
            obs = s[m].mean() / sd[k - 1]
            bs = []
            for _ in range(args.n_boot):
                samp = np.concatenate([rows_of[p] for p in rng.choice(upid, len(upid))])
                sel = samp[m[samp]]
                if len(sel):
                    bs.append(s[sel].mean() / sd[k - 1])
            lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5])
            topic_rows.append({"PC": k, "topic": short_topic_label(t), "n_sentences": int(m.sum()),
                               "position_sd": obs, "ci_low": lo_ci, "ci_high": hi_ci,
                               "excludes_zero": bool(lo_ci > 0 or hi_ci < 0)})
        for end, mask in (("high", hi), ("low", lo)):
            for c in CONDITIONS:
                cm = (df["condition_label"] == c).to_numpy()
                who_rows.append({"PC": k, "end": end, "condition": c,
                                 "share_of_end": (mask & cm).sum() / mask.sum(),
                                 "share_of_all": cm.mean()})
    words = pd.DataFrame(word_rows)
    words.to_csv(out / "pc_end_words.csv", index=False)
    pd.DataFrame(topic_rows).to_csv(out / "pc_end_topic_positions.csv", index=False)
    pd.DataFrame(who_rows).to_csv(out / "pc_end_condition_shares.csv", index=False)

    tp = pd.DataFrame(topic_rows)
    who = pd.DataFrame(who_rows)
    lab_rows = []
    for k in pcs:
        for end in ("high", "low"):
            t = tp[(tp.PC == k) & tp.excludes_zero & ((tp.position_sd > 0) == (end == "high"))]
            t = t.reindex(t.position_sd.abs().sort_values(ascending=False).index)
            w = words[(words.PC == k) & (words.end == end)].reindex(
                words[(words.PC == k) & (words.end == end)].z.abs().sort_values(ascending=False).index).head(6)
            sh = who[(who.PC == k) & (who.end == end)]
            sk = S[:, k - 1]
            mask = sk >= np.percentile(sk, 75) if end == "high" else sk <= np.percentile(sk, 25)
            txt = df["content_text"].fillna("")
            lex = {f"share_{name}_words": txt[mask].str.contains(pat).mean() for name, pat in LEXICONS.items()}
            lab_rows.append({
                **lex,
                "PC": k, "end": end, "label": PC_END_LABELS.get((k, end), ""),
                "topics_CI_excludes_0": "; ".join(f"{r.topic} ({r.position_sd:+.2f} SD, CI {r.ci_low:+.2f}..{r.ci_high:+.2f})"
                                                   for r in t.itertuples()),
                "leading_words_unsupported": "; ".join(f"{r.word} ({GLOSS.get(r.word, '?')}, z={r.z:+.1f})" for r in w.itertuples()),
                "share_of_end_by_condition": "; ".join(f"{r.condition} {100 * r.share_of_end:.0f}% (overall {100 * r.share_of_all:.0f}%)"
                                                        for r in sh.itertuples())})
    pd.DataFrame(lab_rows).to_csv(out / "pc_end_labels.csv", index=False)

    pd.set_option("display.width", 200)
    for k in pcs:
        for end in ("high", "low"):
            w = words[(words.PC == k) & (words.end == end) & words.supported].sort_values("z", key=abs, ascending=False)
            print(f"\\nPC{k} {end} end - supported words ({len(w)}):")
            print(w.head(15)[["word", "z", "count_high", "count_low", "n_participants", "bootstrap_stability"]].round(2).to_string(index=False))
    tp = pd.DataFrame(topic_rows)
    print("\\nTopics whose position CI excludes 0:")
    print(tp[tp.excludes_zero].round(2).to_string(index=False))
    print("\\nCondition shares at the ends:")
    print(pd.DataFrame(who_rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
