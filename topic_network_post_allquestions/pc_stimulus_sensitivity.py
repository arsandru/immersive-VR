#!/usr/bin/env python3
"""Stimulus sensitivity of the PC condition differences (exploratory).

Control waited in a pre-surgery room with no content; VR Only watched nature videos in VR; VR Art experienced an immersive art
installation. Only the VR groups can describe a stimulus, so part of a content difference is expected. This check removes every
sentence that contains a stimulus word (lexicon below, nature video AND art installation vocabulary) and repeats the PC test of
pc_differences.py on what is left.

Procedure: the PCs are those of the full set (PCA fitted on all sentences, so PC4 and PC6 keep their meaning); participant
centroids are recomputed from the remaining sentences and projected onto them (participants with no sentence left drop out);
permutation ANOVA per PC (participant labels shuffled, group sizes kept), Benjamini-Hochberg over the first 10 PCs; Holm-adjusted
permutation tests for the three pairs on PC4 and PC6. The lexicon was written after a first exploratory look at the data (it is not
pre-specified); treat the result as a sensitivity analysis.

Outputs: pc_stimulus_sensitivity.csv, pc_stimulus_dropped.csv (document ids only, no text)
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_network_analysis import CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, bh_fdr, load_data  # noqa: E402
from global_distribution_tests import holm  # noqa: E402
from pc_differences import anova_F  # noqa: E402

STIMULUS_LEXICON = (
    r"v[ií]deos?|filmes?|imagens?|imagem|natureza|[aá]gua|paisagem|paisagens|cen[aá]rios?|floresta|\bmar\b|praia"  # nature videos
    r"|\barte\b|art[ií]stic|\bobras?\b|pintura|exposi[cç]|instala[cç]|\bcores?\b|\bcor\b|m[uú]sica|\bsom\b"  # art installation
)
FOCUS = (4, 6)


def run(S, sd, row_p, lab_all, keep, n_perm, rng, n_pc=10):
    pp = np.unique(row_p[keep])
    P = np.stack([S[keep][row_p[keep] == p].mean(axis=0) for p in pp])[:, :n_pc]
    lab = lab_all[pp]
    F, eta = anova_F(P, lab)
    ge = np.zeros(n_pc)
    for _ in range(n_perm):
        Fp, _ = anova_F(P, rng.permutation(lab))
        ge += Fp >= F - 1e-12
    p = (1 + ge) / (1 + n_perm)
    pair_p = {}
    for k in (f - 1 for f in FOCUS):
        ps = []
        for a, b in combinations(range(3), 2):
            m = np.nonzero(np.isin(lab, (a, b)))[0]
            Xk = P[m][:, [k]]
            f_obs, _ = anova_F(Xk, lab[m])
            cnt = sum(anova_F(Xk, rng.permutation(lab[m]))[0][0] >= f_obs[0] - 1e-12 for _ in range(n_perm))
            ps.append((1 + cnt) / (1 + n_perm))
        pair_p[k] = dict(zip(combinations(range(3), 2), holm(np.array(ps))))
    return dict(P=P, lab=lab, F=F, eta=eta, p=p, p_fdr=bh_fdr(p), pair_p=pair_p, n=[int((lab == g).sum()) for g in range(3)],
                sd_units=P / sd[:n_pc])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)

    df = load_data(args.input, include_outlier=False)
    z = np.load(out / "sentence_embeddings.npz")
    lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
    E = np.stack([lookup[i] for i in df["document_id"]]).astype(np.float64)
    cond_of = df.drop_duplicates("participant_id").set_index("participant_id")["condition_label"]
    pids = list(cond_of.index)
    lab_all = np.array([CONDITIONS.index(cond_of[p]) for p in pids])
    row_p = np.array([pids.index(x) for x in df["participant_id"]])
    pca = PCA(n_components=10, random_state=0).fit(E)
    S = pca.transform(E)
    sd = S.std(axis=0)

    stim = df["sentence"].str.lower().str.contains(STIMULUS_LEXICON, regex=True).to_numpy()
    dropped = df[stim][["document_id", "participant_id", "condition_label", "question_id"]]
    dropped.to_csv(out / "pc_stimulus_dropped.csv", index=False)
    by_c = {c: (int(stim[(df.condition_label == c).to_numpy()].sum()), int((df.condition_label == c).sum())) for c in CONDITIONS}
    left = pd.Series(row_p[~stim]).nunique()
    print(f"stimulus-word sentences: {int(stim.sum())} of {len(df)}; by condition (dropped, total): {by_c}; participants left: {left} of {len(pids)}")

    ref = pd.read_csv(out / "pc_condition_tests.csv")
    rows = []
    for name, keep in (("all sentences", np.ones(len(df), bool)), ("without stimulus-word sentences", ~stim)):
        r = run(S, sd, row_p, lab_all, keep, args.n_perm, rng)
        if name == "all sentences":
            assert np.allclose(r["F"], ref.F.to_numpy(), rtol=1e-6), "does not reproduce pc_condition_tests.csv"
        for k in range(10):
            row = dict(variant=name, PC=k + 1, n_participants=sum(r["n"]), n_VR_Art=r["n"][0], n_VR_Only=r["n"][1], n_Control=r["n"][2],
                       F=r["F"][k], eta_squared=r["eta"][k], p_perm=r["p"][k], p_fdr=r["p_fdr"][k])
            for g, c in enumerate(CONDITIONS):
                row[f"mean_{c}"] = r["sd_units"][r["lab"] == g, k].mean()
            if k + 1 in FOCUS:
                for (a, b), v in r["pair_p"][k].items():
                    row[f"p_holm_{CONDITIONS[a]}_vs_{CONDITIONS[b]}"] = v
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "pc_stimulus_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["variant", "PC", "n_participants", "mean_VR Art", "mean_VR Only", "mean_Control", "F", "eta_squared", "p_perm", "p_fdr"]
    print(res[res.PC.isin(FOCUS)][cols + [c for c in res.columns if c.startswith("p_holm")]].round(4).to_string(index=False))
    print("PCs with p_fdr < .05 without stimulus sentences:", res[(res.variant != "all sentences") & (res.p_fdr < 0.05)].PC.tolist())


if __name__ == "__main__":
    main()
