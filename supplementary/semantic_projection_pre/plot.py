#!/usr/bin/env python3
"""Figure of the condition comparison at ONE timepoint (pre or post; detected from the folder name),
in the layout of the original figure: panels Control, VR Art, VR Only; x = projection on the distress-vs-relief
axis (relief positive), y = PC1 of the residual variance of this timepoint's descriptors; coloured line and band =
model mean with 95% CI; grey band = median absolute score (MAD0).
Outputs semantic_projection_{time}_scatter.pdf/.svg/.png (the model-based headline figure semantic_projection_{time}.pdf comes from models.R)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
TIME = HERE.name.split("_")[-1]
COLORS = {"VR Art": "#8de5a1", "VR Only": "#ffb482", "Control": "#a1c9f4"}
DARK = {"VR Art": "#3fae5d", "VR Only": "#e0823a", "Control": "#5b8fd1"}
ORDER = ["Control", "VR Art", "VR Only"]
SUBSET = "all"


def short(t, n=32):
    return t if len(t) <= n else t[: n - 1].rstrip() + "..."


def main() -> None:
    d = pd.read_csv(HERE / "projection_descriptors.csv")
    d["x"] = d["projection"]
    mad0 = float(np.median(np.abs(d.x)))
    pairs = pd.read_csv(HERE / f"models_descriptor_level_{SUBSET}.csv").set_index("contrast")
    pl = pd.read_csv(HERE / f"models_participant_level_means_{SUBSET}.csv")
    em = pd.read_csv(HERE / f"models_descriptor_means_{SUBSET}.csv").set_index("condition")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharex=True, sharey=True)
    for ax, cond in zip(axes, ORDER):
        sub = d[d.condition_label == cond]
        ax.axvspan(-mad0, mad0, color="#d9d9d9", alpha=0.18, zorder=0)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.scatter(sub.x, sub.y_pc1_residual, s=62, color=COLORS[cond], edgecolor="white", lw=0.5, alpha=0.88, zorder=2)
        lab = pd.concat([sub.nsmallest(5, "x"), sub.nlargest(3, "x")]).drop_duplicates("document_id")
        texts = [ax.text(r.x, r.y_pc1_residual, short(r.sentence), fontsize=8) for r in lab.itertuples()]
        adjust_text(texts, x=lab.x.to_numpy(), y=lab.y_pc1_residual.to_numpy(), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5))
        e = em.loc[cond]
        ax.axvline(e.estimated_mean, color=DARK[cond], lw=2.4, zorder=1)
        ax.axvspan(e.ci_low, e.ci_high, color=DARK[cond], alpha=0.12, zorder=0)
        ax.set_title(f"{cond}  ({sub.participant_id.nunique()} participants, {len(sub)} descriptors)\n"
                     f"model mean {e.estimated_mean:+.3f} [{e.ci_low:+.3f}, {e.ci_high:+.3f}]", fontsize=10.5,
                     color=DARK[cond], fontweight="bold")
        ax.grid(True, alpha=0.3)
    fig.supxlabel("Projection scores on distress-vs-relief axis (relief positive)", fontsize=13)
    axes[0].set_ylabel("PC1 of the residual variance", fontsize=13)
    p3 = ", ".join(f"{k.replace(' vs ', ' v ')} p = {v.p_value:.2f}" for k, v in pairs.iterrows())
    fig.suptitle(f"{TIME.capitalize()}-interview descriptors: projection by condition. Coloured line and band = model mean "
                 f"with 95% CI; grey band = MAD0.\nMain model, descriptor-level mixed model (CR2): {p3}; sensitivity, participant-level F-test p = {pl.F_p.iloc[0]:.2f}",
                 fontsize=10.2, y=1.02)
    fig.tight_layout()
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 250})):
        fig.savefig(HERE / f"semantic_projection_{TIME}_scatter.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote", HERE / f"semantic_projection_{TIME}_scatter.[pdf|svg|png]")


if __name__ == "__main__":
    main()
