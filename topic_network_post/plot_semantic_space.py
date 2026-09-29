#!/usr/bin/env python3
"""Figures for the topic-free embedding-space analysis.

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

from all_questions.topic_labels import short_topic_label  # noqa: E402
from q3.plot_topic_differences import CONDITION_COLORS, CONDITION_ORDER  # noqa: E402
from topic_network_analysis import DEFAULT_INPUT, DEFAULT_OUTPUT, load_data  # noqa: E402

DARK = {"VR Art": "#3fae5d", "VR Only": "#e0823a", "Control": "#5b8fd1"}


def save(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {base}.[pdf|png]")


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
    """One row per PC: group means, test result, and the topics at the ends of the PC."""
    res = pd.read_csv(out / "pc_condition_tests.csv")
    ld = pd.read_csv(out / "pc_topic_loadings.csv", index_col=0)
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
    vmax = np.abs(ld.values).max()
    im = ax_h.imshow(ld.values.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, zorder=2)
    for i, r in res.iterrows():
        if r.p_fdr < 0.05:
            ax_h.add_patch(plt.Rectangle((-0.5, i - 0.5), ld.shape[0], 1, fill=False, ec="#222222", lw=1.6, zorder=4))
    ax_h.set_xticks(range(ld.shape[0]))
    ax_h.set_xticklabels(ld.index, rotation=40, ha="left", fontsize=10)
    ax_h.xaxis.tick_top()
    ax_h.set_yticks([])
    ax_h.tick_params(length=0)
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(K - 0.5, -0.5)
    ax_t.axis("off")
    for i, r in res.iterrows():
        col = ld[f"PC{int(r.PC)}"].sort_values()
        star = "*" if r.p_fdr < 0.05 else ""
        ax_t.text(0.0, i - 0.16, f"F = {r.F:.1f}   eta² = {r.eta_squared:.2f}   p(FDR) = {r.p_fdr:.3f}{star}",
                  fontsize=9.5, va="center", fontweight="bold" if star else "normal")
        ax_t.text(0.0, i + 0.20, f"high: {', '.join(col.index[::-1][:2])}   |   low: {', '.join(col.index[:2])}",
                  fontsize=8.3, va="center", color="#444444")
    fig.colorbar(im, ax=ax_h, orientation="horizontal", fraction=0.035, pad=0.02, shrink=0.5,
                 label="where each topic sits on the PC, in SD (red = high end, blue = low end)")
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
    ld = pd.read_csv(out / "pc_topic_loadings.csv", index_col=0)
    sig = res[res.p_fdr < 0.05].reset_index(drop=True)
    lab = pd.read_csv(out / "pc_end_labels.csv")

    def end_label(pc, end, col):
        r = lab[(lab.PC == pc) & (lab.end == end)]
        text = r["label"].iloc[0] if len(r) and isinstance(r["label"].iloc[0], str) and r["label"].iloc[0] else None
        return text or (col.index[-1] if end == "high" else col.index[0])

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
        col = ld[f"PC{int(r.PC)}"].sort_values()
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
        box.text(0.535, yc + 0.09, "high: " + end_label(int(r.PC), "high", col), fontsize=8.2, va="center")
        box.text(0.535, yc - 0.09, "low:  " + end_label(int(r.PC), "low", col), fontsize=8.2, va="center")
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
    topics = sorted(df["Topic"].unique())
    tcent = {t: E[(df["Topic"] == t).to_numpy()].mean(axis=0) for t in topics}
    T2 = {t: pca.transform(v[None, :])[0][AX] for t, v in tcent.items()}
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
    for t in topics:
        ax.text(*T2[t], short_topic_label(t), fontsize=8.5, color="#444444", ha="center", va="center",
                style="italic", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    ax.set_xlabel(f"PC{AX[0] + 1} ({100 * ev[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC{AX[1] + 1} ({100 * ev[1]:.1f}% of variance)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#eeeeee")
    ax.set_axisbelow(True)
    t1 = content_p("VR Art vs Control")
    ax.set_title("Semantic space of all sentences\n"
                 "small dots = sentences, dots = participants, diamond + ellipse = condition mean "
                 "with 95% bootstrap region; axes = the two PCs that separate the conditions best (picked for display, tests use the full embedding)\n"
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
