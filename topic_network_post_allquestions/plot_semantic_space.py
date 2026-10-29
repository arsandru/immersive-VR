#!/usr/bin/env python3
"""Figures for the topic-free embedding-space analysis (no topic model: PCs are read from sentences, see pc_sentence_interpretation.py).

  semantic_space_by_condition.*   PCA map of the sentence embeddings with condition centroids,
                                  plus a heat table of the content tests
  embedding_space_tests.*         p-value table for the content tests
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from q3.plot_topic_differences import CONDITION_COLORS, CONDITION_ORDER  # noqa: E402
from topic_network_analysis import DEFAULT_INPUT, DEFAULT_OUTPUT, load_data  # noqa: E402

DARK = {"VR Art": "#3fae5d", "VR Only": "#e0823a", "Control": "#5b8fd1"}


def save(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {base}.[pdf|png]")


def rep_words(words, pc, end, n=4):
    """Representative words of one PC end: the n lemmas with the strongest evidence in that direction, ranked by the association
    (SD difference) divided by its bootstrap standard error, among lemmas whose participant-cluster bootstrap CI excludes 0. This
    favours words that are both strongly associated and used by many participants over rare words with large raw differences."""
    w = words[(words.PC == pc) & ((words.ci_low > 0) | (words.ci_high < 0))].copy()
    w["z"] = w.mean_diff_sd / ((w.ci_high - w.ci_low) / 3.92)
    w = w[w.z > 0].nlargest(n, "z") if end == "high" else w[w.z < 0].nsmallest(n, "z")
    return ", ".join(w.word)


def ellipse(ax, pts, color, n_std2=5.991):
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w, h = 2 * np.sqrt(n_std2 * vals[::-1])
    ax.add_patch(Ellipse(mu, w, h, angle=ang, facecolor=color, edgecolor=DARK[color_name(color)],
                         alpha=0.28, lw=1.5, zorder=3))


def color_name(c):
    return next(k for k, v in CONDITION_COLORS.items() if v == c)


def stars_text(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


SHORT_TEST = {"sentence PERMANOVA (cosine), pseudo-F": "Sentence-level\nPERMANOVA",
              "participant-centroid PERMANOVA, pseudo-F": "Participant-level\nPERMANOVA",
              "dispersion of participant centroids, F": "Dispersion\n(spread)"}
COLS = [("all three", "All three"), ("VR Art vs VR Only", "VR Art\nvs VR Only"),
        ("VR Art vs Control", "VR Art\nvs Control"), ("VR Only vs Control", "VR Only\nvs Control")]


def draw_test_table(ax, tests, short=True, fs=11):
    """Heat table of permutation p-values (omnibus: raw p; pairs: Holm-adjusted p)."""
    cmap = LinearSegmentedColormap.from_list("p", ["#c0392b", "#f5b7a5", "#ffffff"])
    rows_ = list(dict.fromkeys(tests.test))
    for i, rname in enumerate(rows_):
        for j, (c, _) in enumerate(COLS):
            rr = tests[(tests.test == rname) & (tests.comparison == c)].iloc[0]
            val = rr.p_holm if (c != "all three" and pd.notna(rr.p_holm)) else rr.p_perm
            ax.add_patch(plt.Rectangle((j, len(rows_) - 1 - i), 1, 1, color=cmap(np.clip(val / 0.2, 0, 1)),
                                       ec="white", lw=2))
            ax.text(j + 0.5, len(rows_) - 1 - i + 0.5, f"{val:.3f}" + ("*" if val < 0.05 else ""),
                    ha="center", va="center", fontsize=fs, fontweight="bold" if val < 0.05 else "normal")
    ax.set_xlim(0, len(COLS))
    ax.set_ylim(0, len(rows_))
    ax.set_xticks(np.arange(len(COLS)) + 0.5)
    ax.set_xticklabels([l for _, l in COLS], fontsize=fs - 0.5)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows_)) + 0.5)
    ax.set_yticklabels([SHORT_TEST.get(r, r) if short else r for r in rows_[::-1]], fontsize=fs - 0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_pc_profile(out):
    """One row per PC: group means, test result, sentence-based label for the PCs that pass FDR, and a word matrix."""
    res = pd.read_csv(out / "pc_condition_tests.csv")
    words = pd.read_csv(out / "pc_sentence_words.csv")
    K = len(res)
    fig = plt.figure(figsize=(17, 0.62 * K + 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 4.6, 2.6], wspace=0.04)
    ax_d, ax_h, ax_t = (fig.add_subplot(gs[i]) for i in range(3))
    offs = {"VR Art": -0.24, "VR Only": 0.0, "Control": 0.24}
    for i, r in res.iterrows():
        if r.p_fdr < 0.05:
            for a in (ax_d, ax_h, ax_t):
                a.axhspan(i - 0.5, i + 0.5, color="#fff4cc", zorder=0, lw=0)
        for c in CONDITION_ORDER:
            m, lo, hi = r[f"mean_{c}"], r[f"ci_low_{c}"], r[f"ci_high_{c}"]
            ax_d.errorbar(m, i + offs[c], xerr=[[m - lo], [hi - m]], fmt="o", ms=6.5, color=CONDITION_COLORS[c],
                          mec="white", mew=0.6, ecolor=CONDITION_COLORS[c], elinewidth=1.4, capsize=2, zorder=3)
    ax_d.axvline(0, color="#777777", lw=0.9, zorder=1)
    ax_d.set_ylim(K - 0.5, -0.5)
    ax_d.set_yticks(range(K))
    ax_d.set_yticklabels([f"PC{int(r.PC)}  ({100 * r.variance_explained:.1f}%)" for _, r in res.iterrows()],
                         fontsize=11)
    ax_d.set_xlabel("Group mean score (SD units of the PC), 95% bootstrap CI")
    ax_d.xaxis.set_label_position("top")
    ax_d.xaxis.tick_top()
    ax_d.grid(axis="x", color="#e6e6e6")
    ax_d.set_axisbelow(True)
    for sp in ("right", "bottom"):
        ax_d.spines[sp].set_visible(False)
    # word matrix (sentence-based): difference in mean PC score (SD) between sentences containing the word and those that do not.
    # Columns = the words that go most strongly with each end of the PCs that pass FDR (bootstrap CI excluding 0; up to 6 per end, ranked by association / bootstrap SE);
    # dots mark cells whose participant-cluster bootstrap 95% CI excludes 0.
    sig_pcs = [int(p) for p in res.PC[res.p_fdr < 0.05]]
    cols = []
    for pc in sig_pcs:
        w = words[(words.PC == pc) & ((words.ci_low > 0) | (words.ci_high < 0))].copy()
        w["z"] = w.mean_diff_sd / ((w.ci_high - w.ci_low) / 3.92)  # same ranking as rep_words
        for part in (w[w.z > 0].nlargest(6, "z"), w[w.z < 0].nsmallest(6, "z")):
            cols += [x for x in part.word if x not in cols]
    ref = words[words.PC == sig_pcs[0]].set_index("word").mean_diff_sd
    cols = sorted(cols, key=lambda x: ref[x])
    D = words.pivot(index="PC", columns="word", values="mean_diff_sd").loc[:, cols]
    Fl = words.assign(sig=(words.ci_low > 0) | (words.ci_high < 0)).pivot(index="PC", columns="word", values="sig").loc[:, cols]
    im = ax_h.imshow(D.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, zorder=2)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if Fl.values[i, j]:
                ax_h.plot(j, i, "o", ms=3.2, color="#222222", zorder=5)
    for i, r in res.iterrows():
        if r.p_fdr < 0.05:
            ax_h.add_patch(plt.Rectangle((-0.5, i - 0.5), D.shape[1], 1, fill=False, ec="#222222", lw=1.6, zorder=4))
    ax_h.set_xticks(range(D.shape[1]))
    ax_h.set_xticklabels(D.columns, rotation=45, ha="left", fontsize=9)
    ax_h.xaxis.tick_top()
    ax_h.set_yticks([])
    ax_h.tick_params(length=0)
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(K - 0.5, -0.5)
    ax_t.axis("off")
    lab = pd.read_csv(out / "pc_end_labels.csv")  # sentence-based labels (pc_end_evidence.py), written for the PCs that pass FDR
    for i, r in res.iterrows():
        star = "*" if r.p_fdr < 0.05 else ""
        ax_t.text(0.0, i - 0.26 if star else i, f"F = {r.F:.1f}   eta² = {r.eta_squared:.2f}   p(FDR) = {r.p_fdr:.3f}{star}",
                  fontsize=9.5, va="center", fontweight="bold" if star else "normal")
        ends = {e: lab[(lab.PC == int(r.PC)) & (lab.end == e)] for e in ("high", "low")}
        if star and all(len(v) for v in ends.values()):  # only PCs that separate the conditions are interpreted
            for dy, e in ((0.06, "high"), (0.30, "low")):
                ax_t.text(0.0, i + dy, f"{e}: {ends[e].label.iloc[0]} ({rep_words(words, int(r.PC), e)})",
                          fontsize=7.6, va="center", color="#444444")
    fig.colorbar(im, ax=ax_h, orientation="horizontal", fraction=0.035, pad=0.02, shrink=0.5,
                 label="words that go with the ends of the labelled PCs: sentences containing the word minus those without, in SD of the PC (dot: bootstrap CI excludes 0)")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", ls="", color=CONDITION_COLORS[c], mec="white", ms=9)
                for c in CONDITION_ORDER], CONDITION_ORDER, loc="lower left", bbox_to_anchor=(0.06, 0.02),
               ncol=3, frameon=False, fontsize=11)
    fig.suptitle("Which principal components separate the conditions, and what do they point toward?\n"
                 "yellow/outlined rows: omnibus permutation test significant after Benjamini-Hochberg over the "
                 f"first {K} PCs; PC signs are arbitrary", fontsize=11.5, y=0.995)
    fig.subplots_adjust(top=0.82, bottom=0.1)
    save(fig, out / "pc_differences_by_condition")


def draw_pc_inset(ax, out):
    """Bordered box next to the heat table: the PCs that separate the conditions and what they point toward."""
    res = pd.read_csv(out / "pc_condition_tests.csv")
    sig = res[res.p_fdr < 0.05].reset_index(drop=True)
    lab = pd.read_csv(out / "pc_end_labels.csv")

    def end_label(pc, end):
        r = lab[(lab.PC == pc) & (lab.end == end)]
        return r["label"].iloc[0] if len(r) and isinstance(r["label"].iloc[0], str) else ""

    box = ax.inset_axes([0.398, 0.018, 0.375, 0.165])
    box.set_zorder(10)
    box.set_xticks([])
    box.set_yticks([])
    box.set_xlim(0, 1)
    box.set_ylim(0, 1)
    box.set_facecolor("white")
    for sp in box.spines.values():
        sp.set_edgecolor("#555555")
        sp.set_linewidth(1.3)
    box.text(0.5, 0.93, f"PCs that separate the conditions (FDR over first {len(res)} PCs)",
             ha="center", va="center", fontsize=8.2, fontweight="bold")
    n = max(len(sig), 1)
    row_h = 0.74 / n
    for i, r in sig.iterrows():
        yc = 0.845 - (i + 0.5) * row_h
        box.text(0.015, yc + 0.09, f"PC{int(r.PC)}", fontsize=9, fontweight="bold", va="center")
        box.text(0.015, yc - 0.09, f"{100 * r.variance_explained:.1f}% var., p(FDR) {r.p_fdr:.3f}", fontsize=7,
                 va="center", color="#333333")
        d = box.inset_axes([0.285, yc - 0.32 * row_h, 0.22, 0.64 * row_h])
        for k, c in enumerate(CONDITION_ORDER):
            m, lo, hi = r[f"mean_{c}"], r[f"ci_low_{c}"], r[f"ci_high_{c}"]
            d.errorbar(m, k, xerr=[[m - lo], [hi - m]], fmt="o", ms=4.6, color=CONDITION_COLORS[c], mec="white",
                       mew=0.4, ecolor=CONDITION_COLORS[c], elinewidth=1.1, capsize=1.5)
        d.axvline(0, color="#888888", lw=0.7)
        d.set_ylim(2.6, -0.6)
        d.set_xlim(-0.75, 0.75)
        d.set_yticks([])
        d.set_xticks([])
        for sp in ("left", "right", "top"):
            d.spines[sp].set_visible(False)
        box.text(0.535, yc + 0.09, "high: " + end_label(int(r.PC), "high"), fontsize=8.2, va="center")
        box.text(0.535, yc - 0.09, "low:  " + end_label(int(r.PC), "low"), fontsize=8.2, va="center")
    box.text(0.5, 0.035, "dots = group means in SD units of the PC (bars: 95% bootstrap CI, line = 0)",
             fontsize=6.4, color="#666666", va="center", ha="center")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(3)

    df = load_data(args.input, include_outlier=False)
    z = np.load(out / "sentence_embeddings.npz")
    lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
    E = np.stack([lookup[i] for i in df["document_id"]]).astype(np.float64)
    tests = pd.read_csv(out / "embedding_space_tests.csv")

    cond_of = df.drop_duplicates("participant_id").set_index("participant_id")["condition_label"]
    pids = list(cond_of.index)
    row_p = np.array([pids.index(x) for x in df["participant_id"]])
    cent = np.stack([E[row_p == i].mean(axis=0) for i in range(len(pids))])
    cond = np.array([cond_of[x] for x in pids])

    # axes = the two PCs that separate the conditions most (largest eta^2 among those passing the FDR test)
    pct = pd.read_csv(out / "pc_condition_tests.csv")
    AX = (pct[pct.p_fdr < 0.05].sort_values("eta_squared", ascending=False).PC.astype(int).head(2) - 1).tolist()
    pca = PCA(n_components=max(AX) + 1, random_state=0).fit(E)
    S, C2 = pca.transform(E)[:, AX], pca.transform(cent)[:, AX]
    ev = pca.explained_variance_ratio_[AX]

    def content_p(comp):
        r = tests[tests.test.str.startswith("participant-centroid") & (tests.comparison == comp)]
        return r.iloc[0]

    fig = plt.figure(figsize=(13, 9.6))
    ax = fig.add_axes([0.065, 0.06, 0.925, 0.86])
    for c in CONDITION_ORDER:
        m = (df["condition_label"] == c).to_numpy()
        ax.scatter(S[m, 0], S[m, 1], s=16, color=CONDITION_COLORS[c], alpha=0.35, lw=0, zorder=1)
    for c in CONDITION_ORDER:
        m = cond == c
        ax.scatter(C2[m, 0], C2[m, 1], s=46, color=CONDITION_COLORS[c], edgecolor="white", lw=0.8,
                   zorder=4)
        boots = np.stack([C2[m][rng.integers(0, m.sum(), m.sum())].mean(axis=0)
                          for _ in range(args.n_boot)])
        ellipse(ax, boots, CONDITION_COLORS[c])
        mu = C2[m].mean(axis=0)
        ax.scatter(*mu, s=150, marker="D", color=DARK[c], edgecolor="white", lw=1.5, zorder=6,
                   label=f"{c} (participant mean)")
    # representative words of the PC ends, drawn at the mean position of the sentences that contain them
    wpos = pd.read_csv(out / "pc_sentence_word_positions.csv").set_index("word")
    wl = pd.read_csv(out / "pc_sentence_words.csv")
    shown = []
    for k in AX:
        for end in ("high", "low"):
            shown += [w for w in rep_words(wl, k + 1, end).split(", ") if w and w not in shown]
    texts = [ax.text(wpos.loc[w, f"PC{AX[0] + 1}"], wpos.loc[w, f"PC{AX[1] + 1}"], w, fontsize=9.5, color="#333333", ha="center",
                     va="center", style="italic", zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
             for w in shown]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.2, 1.4))
    except ImportError:
        pass
    ax.set_xlabel(f"PC{AX[0] + 1} ({100 * ev[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC{AX[1] + 1} ({100 * ev[1]:.1f}% of variance)")
    lab_ = pd.read_csv(out / "pc_end_labels.csv")  # sentence-based labels of the PC ends (pc_end_evidence.py)

    words_ = pd.read_csv(out / "pc_sentence_words.csv")

    def end_lab(pc, end):
        r = lab_[(lab_.PC == pc) & (lab_.end == end)]
        return (r["label"].iloc[0] + "\n" + rep_words(words_, pc, end)) if len(r) else ""
    ax.text(0.0, -0.062, f"\u2190 {end_lab(AX[0] + 1, 'low')}", transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#333333")
    ax.text(1.0, -0.062, f"{end_lab(AX[0] + 1, 'high')} \u2192", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#333333")
    ax.text(-0.052, 0.0, f"\u2190 {end_lab(AX[1] + 1, 'low')}", transform=ax.transAxes, ha="right", va="bottom", rotation=90, fontsize=9.5, color="#333333")
    ax.text(-0.052, 1.0, f"{end_lab(AX[1] + 1, 'high')} \u2192", transform=ax.transAxes, ha="right", va="top", rotation=90, fontsize=9.5, color="#333333")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#eeeeee")
    ax.set_axisbelow(True)
    t1 = content_p("VR Art vs Control")
    ax.set_title("Semantic space of all sentences\n"
                 "small dots = sentences, dots = participants, diamond + ellipse = condition mean "
                 "with 95% bootstrap region; axes = the two PCs that separate the conditions best (picked for display, tests use the full embedding)\n"
                 "italic words = representative words of the PC ends, at the mean position of the sentences that contain them\n"
                 f"VR Art vs Control (participant-level PERMANOVA): p = {t1.p_perm:.3f}, "
                 f"Holm {t1.p_holm:.3f}", fontsize=10.5)

    # ---- inset (lower left): heat table of the content tests
    ymin, ymax = S[:, 1].min(), S[:, 1].max()
    ax.set_ylim(ymin - 0.30 * (ymax - ymin), ymax + 0.03 * (ymax - ymin))
    box = ax.inset_axes([0.012, 0.018, 0.375, 0.165])  # bordered container
    box.set_zorder(10)
    box.set_xticks([])
    box.set_yticks([])
    box.set_facecolor("white")
    for sp in box.spines.values():
        sp.set_edgecolor("#555555")
        sp.set_linewidth(1.3)
    axt = box.inset_axes([0.325, 0.03, 0.66, 0.60])
    draw_test_table(axt, tests, fs=8.5)
    draw_pc_inset(ax, out)
    save(fig, out / "semantic_space_by_condition")

    draw_pc_profile(out)

    # ---- stand-alone table
    fig, ax = plt.subplots(figsize=(9, 3.6))
    draw_test_table(ax, tests, short=False)
    ax.set_title("Embedding-space content tests (participant-label permutation). "
                 "All three: raw p; pairs: Holm-adjusted p. * = p < .05", fontsize=10, y=1.25)
    save(fig, out / "embedding_space_tests")

if __name__ == "__main__":
    main()
