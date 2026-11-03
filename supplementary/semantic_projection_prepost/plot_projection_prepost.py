#!/usr/bin/env python3
"""Figures for the pre vs post semantic projection.

  semantic_projection_prepost_scatter.*  descriptor scatter, rows pre / post, columns Control / VR Art / VR Only
                                         (x = projection on the distress-vs-relief axis, y = PC1 of the residual,
                                         as in the original figure); most extreme descriptors labelled
  semantic_projection_prepost_change.*   participant means pre -> post per condition, with condition means (95% CI)
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
COLORS = {"VR Art": "#8de5a1", "VR Only": "#ffb482", "Control": "#a1c9f4"}
DARK = {"VR Art": "#3fae5d", "VR Only": "#e0823a", "Control": "#5b8fd1"}
ORDER = ["Control", "VR Art", "VR Only"]  # same panel order as the original figure
TAG = "all"


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.svg", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {HERE / name}.[pdf|svg|png]")


def short(t, n=34):
    return t if len(t) <= n else t[: n - 1].rstrip() + "..."


def scatter(d):
    mad0 = float(np.median(np.abs(d.projection)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for r, time in enumerate(("pre", "post")):
        for c, cond in enumerate(ORDER):
            ax = axes[r, c]
            sub = d[(d.condition_label == cond) & (d.time == time)]
            ax.axvspan(-mad0, mad0, color="#d9d9d9", alpha=0.18, zorder=0)
            ax.axvline(0, color="gray", lw=0.8, ls="--")
            ax.scatter(sub.projection, sub.y_pc1_residual, s=60, color=COLORS[cond], edgecolor="white", lw=0.5,
                       alpha=0.85, zorder=2)
            lab = pd.concat([sub.nsmallest(3, "projection"), sub.nlargest(3, "projection")]).drop_duplicates("document_id")
            texts = [ax.text(row.projection, row.y_pc1_residual, short(row.sentence), fontsize=8)
                     for row in lab.itertuples()]
            adjust_text(texts, x=lab.projection.to_numpy(), y=lab.y_pc1_residual.to_numpy(), ax=ax,
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5))
            m = sub.projection.mean()
            ax.axvline(m, color=DARK[cond], lw=2.2, alpha=0.9, zorder=1)
            ax.set_title(f"{cond} - {time}   (n = {len(sub)} descriptors, mean {m:+.3f})", fontsize=11.5,
                         color=DARK[cond], fontweight="bold")
            ax.grid(True, alpha=0.3)
    fig.supxlabel("Projection scores on distress-vs-relief axis (relief positive)", fontsize=13)
    fig.supylabel("PC1 of the residual variance", fontsize=13)
    fig.suptitle("Semantic projection of Q1 descriptors before (top) and after (bottom) VR; "
                 "vertical coloured line = mean; grey band = median absolute score (MAD0)", fontsize=11.5, y=0.995)
    fig.tight_layout()
    save(fig, "semantic_projection_prepost_scatter")


def change():
    p = pd.read_csv(HERE / f"participant_change_{TAG}.csv")
    ps = pd.read_csv(HERE / f"models_participant_paired_{TAG}.csv").set_index("condition")
    ct = pd.read_csv(HERE / f"models_contrasts_{TAG}.csv").set_index("contrast")
    chg = pd.read_csv(HERE / f"models_participant_change_by_condition_{TAG}.csv")
    p_inter = float(chg["Pr(>F)"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, cond in zip(axes, ORDER):
        s = p[p.condition == cond]
        for _, r in s.iterrows():
            ax.plot([0, 1], [r["y.pre"], r["y.post"]], color=COLORS[cond], lw=1.1, alpha=0.7, zorder=1)
        ax.scatter(np.zeros(len(s)), s["y.pre"], s=32, facecolor="white", edgecolor=COLORS[cond], lw=1.2, zorder=2)
        ax.scatter(np.ones(len(s)), s["y.post"], s=36, color=COLORS[cond], zorder=2)
        row = ps.loc[cond]
        ax.plot([0, 1], [row.mean_pre, row.mean_post], color=DARK[cond], lw=3.6, zorder=4)
        ax.scatter([0], [row.mean_pre], s=150, marker="D", facecolor="white", edgecolor=DARK[cond], lw=2.2, zorder=5)
        ax.scatter([1], [row.mean_post], s=150, marker="D", color=DARK[cond], edgecolor="white", lw=1.4, zorder=5)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        drow = ct.loc[f"{cond}: post - pre"]
        sig = "***" if drow.p_value < 0.001 else "**" if drow.p_value < 0.01 else "*" if drow.p_value < 0.05 else "n.s."
        ax.set_title(f"{cond}  (n = {int(row.n)} participants)\nmixed model change {drow.estimate:+.3f} [{drow.ci_low:+.2f}, {drow.ci_high:+.2f}]\n"
                     f"p = {drow.p_value:.3f} {sig}  (FDR-adjusted {drow.p_fdr_within_family:.2f})\n"
                     f"participant-level: {row.mean_change:+.3f}, p = {row.p_t:.3f}",
                     fontsize=9.6, color=DARK[cond], fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pre", "post"], fontsize=12)
        ax.set_xlim(-0.35, 1.35)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Participant mean projection (relief positive)", fontsize=12)
    pl = ct.loc["VR (Art + Only) vs Control: change difference"]
    ia, io = ct.loc["VR Art vs Control: change difference"], ct.loc["VR Only vs Control: change difference"]
    fig.suptitle("Change in distress-vs-relief projection, pre -> post (descriptor-level mixed model, CR2; lines = participant means)\n"
                 f"does the change differ by condition?  VR Art vs Control {ia.estimate:+.3f} (p = {ia.p_value:.2f});  "
                 f"VR Only vs Control {io.estimate:+.3f} (p = {io.p_value:.2f});  planned VR vs Control {pl.estimate:+.3f} "
                 f"[{pl.ci_low:+.3f}, {pl.ci_high:+.3f}], p = {pl.p_value:.2f};  sensitivity, participant-level F-test p = {p_inter:.2f}",
                 fontsize=10.0, y=1.06)
    fig.tight_layout()
    save(fig, "semantic_projection_prepost_change")


if __name__ == "__main__":
    d = pd.read_csv(HERE / "projection_descriptors.csv")
    scatter(d)
    change()
