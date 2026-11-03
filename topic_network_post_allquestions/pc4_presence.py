#!/usr/bin/env python3
"""Does a participant's mean PC4 score go with their presence score? (exploratory)

PC4 is the PC of the full-interview sentence-embedding space that separates the conditions (high = "Experience & senses",
low = "Procedure & care"; see README). Presence is the `presence` column (0-10) of data/anxiety_fear_mechanism_items.csv,
whose `participant` column (P0001, ...) is the interview participant number, checked against data/participant_conditions.csv:
the condition label must agree for every participant.

Primary test (one, so no multiplicity correction): Spearman correlation between the participant's mean PC4 score (PCA fitted on
all sentences, as in pc_differences.py; participant mean of sentence scores) and presence, pooled over all participants with both
values; permutation p (presence shuffled across participants, 10000, two-sided), participant bootstrap 95% CI.
Confound checks (sensitivity analyses, not tests of new hypotheses; reported whatever they show): condition is a confound (Control has
lower presence and lower PC4), so the correlation is repeated
 - within group: Spearman correlation of the residuals after removing group means (a partial rank correlation with group as the
   covariate); permutation p shuffles presence within each group; stratified bootstrap CI.
 - per group: Spearman rho, n, bootstrap CI (descriptive).
Presence has a floor (many zeros, mostly Control): the number of zeros per group is reported.

Outputs: pc4_presence.csv, pc4_presence.[pdf|png]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_network_analysis import CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, load_data  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
PC = 4
COND_MAP = {"Control (care as usual)": "Control", "Creation VR (VR Art)": "VR Art", "Nature VR (VR Only)": "VR Only"}


def spearman(x, y):
    return np.corrcoef(rankdata(x), rankdata(y))[0, 1]


def resid(v, lab):
    r = v.astype(float).copy()
    for g in np.unique(lab):
        r[lab == g] -= r[lab == g].mean()
    return r


def within_rho(x, y, lab):
    return spearman(resid(x, lab), resid(y, lab))


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
    S = PCA(n_components=10, random_state=0).fit_transform(E)
    sd = S.std(axis=0)
    P = np.stack([S[row_p == i].mean(axis=0) for i in range(len(pids))])

    ref = pd.read_csv(out / "pc_condition_tests.csv").set_index("PC")
    for g, c in enumerate(CONDITIONS):  # reproduces the PC4 group means of the existing analysis (same sign)
        assert abs((P[lab_all == g, PC - 1] / sd[PC - 1]).mean() - ref.loc[PC, f"mean_{c}"]) < 1e-6, "PC4 does not reproduce"

    pr = pd.read_csv(DATA / "anxiety_fear_mechanism_items.csv")
    pr["participant_id"] = pr["participant"].str.extract(r"(\d+)")[0].astype(int)
    pr["condition_label"] = pr["condition"].map(COND_MAP)
    pc = pd.read_csv(DATA / "participant_conditions.csv").dropna(subset=["condition_number"])
    pc["cond"] = pc["condition_number"].map({1: "VR Art", 2: "VR Only", 3: "Control"})
    chk = pr.merge(pc, left_on="participant_id", right_on="enrollment_number")
    assert len(chk) == len(pr) and (chk["cond"] == chk["condition_label"]).all(), "presence file and interview conditions disagree"

    d = pd.DataFrame({"participant_id": pids, "condition": [CONDITIONS[g] for g in lab_all], "pc4": P[:, PC - 1] / sd[PC - 1]})
    d = d.merge(pr[["participant_id", "presence"]], on="participant_id", how="left")
    print(f"interviewed participants with PC4: {len(d)}; with presence: {int(d.presence.notna().sum())}")
    d = d.dropna(subset=["presence"]).reset_index(drop=True)
    x, y = d["pc4"].to_numpy(), d["presence"].to_numpy()
    lab = np.array([CONDITIONS.index(c) for c in d["condition"]])
    n = len(d)
    print("n per condition:", {c: int((lab == g).sum()) for g, c in enumerate(CONDITIONS)})
    print("presence = 0 per condition:", {c: int(((y == 0) & (lab == g)).sum()) for g, c in enumerate(CONDITIONS)})

    def boot(stat, strata=False, groups=None):
        vals = np.empty(args.n_perm)
        for b in range(args.n_perm):
            if strata:
                idx = np.concatenate([rng.choice(np.nonzero(lab == g)[0], (lab == g).sum()) for g in range(3)])
            else:
                idx = rng.integers(0, n, n)
            vals[b] = stat(idx)
        return np.nanpercentile(vals, [2.5, 97.5])

    rows = []
    r_obs = spearman(x, y)
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(args.n_perm)])
    lo, hi = boot(lambda i: spearman(x[i], y[i]))
    rows.append(dict(analysis="PRIMARY: pooled", n=n, rho=r_obs, ci_low=lo, ci_high=hi, p_perm=(1 + (abs(null) >= abs(r_obs) - 1e-12).sum()) / (1 + args.n_perm)))

    rw = within_rho(x, y, lab)
    nullw = np.empty(args.n_perm)
    for b in range(args.n_perm):
        yp = y.copy()
        for g in range(3):
            m = np.nonzero(lab == g)[0]
            yp[m] = rng.permutation(y[m])
        nullw[b] = within_rho(x, yp, lab)
    lo, hi = boot(lambda i: within_rho(x[i], y[i], lab[i]), strata=True)
    rows.append(dict(analysis="confound check: within group (group means removed)", n=n, rho=rw, ci_low=lo, ci_high=hi,
                     p_perm=(1 + (abs(nullw) >= abs(rw) - 1e-12).sum()) / (1 + args.n_perm)))

    for g, c in enumerate(CONDITIONS):
        m = np.nonzero(lab == g)[0]
        rg = spearman(x[m], y[m])
        ng = np.array([spearman(x[m], rng.permutation(y[m])) for _ in range(args.n_perm)])
        bs = np.array([spearman(x[m][i], y[m][i]) for i in (rng.integers(0, len(m), len(m)) for _ in range(args.n_perm))])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(analysis=f"confound check: {c} only (descriptive)", n=len(m), rho=rg, ci_low=lo, ci_high=hi,
                         p_perm=(1 + (abs(ng) >= abs(rg) - 1e-12).sum()) / (1 + args.n_perm)))
    res = pd.DataFrame(rows)
    res.to_csv(out / "pc4_presence.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))

    cols = {"VR Art": "#c0392b", "VR Only": "#2471a3", "Control": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    jit = rng.uniform(-0.12, 0.12, n)
    for c in CONDITIONS:
        m = (d["condition"] == c).to_numpy()
        ax.scatter(x[m], y[m] + jit[m], s=34, color=cols[c], alpha=0.85, label=f"{c} (n = {int(m.sum())})", edgecolor="white", lw=0.5)
    ax.set_xlabel("mean PC4 score (SD units; high = Experience & senses, low = Procedure & care)")
    ax.set_ylabel("presence (0-10, jittered)")
    p0, pw = res.iloc[0], res.iloc[1]
    ax.set_title(f"Spearman rho = {p0.rho:.2f} [{p0.ci_low:.2f}, {p0.ci_high:.2f}], p = {p0.p_perm:.3f}; within group {pw.rho:.2f} "
                 f"[{pw.ci_low:.2f}, {pw.ci_high:.2f}], p = {pw.p_perm:.3f}\nexploratory", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "pc4_presence.pdf", bbox_inches="tight")
    fig.savefig(out / "pc4_presence.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
