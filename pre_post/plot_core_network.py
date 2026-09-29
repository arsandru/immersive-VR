#!/usr/bin/env python3
"""Core networks of the pre/post Q1 descriptors (same design as topic_network_post/plot_topic_network.py, scope
"within response": transitions between consecutive descriptors of one participant's Q1 list).

  core_network_within_response_pre.*   pre interviews, one panel per condition
  core_network_pre_vs_post.*           pre (top row) vs post (bottom row), one column per condition

Per panel: PageRank of the topic flow graph (damping 0.85, dangling topics jump uniformly) over all participants of the
condition at that timepoint; the most central topic sits in the middle, the next three on an inner ring, the rest on an
outer ring (angular order follows the embedding MDS of the topic centroids); node size and colour = PageRank (one
colour and size scale for all panels of a figure); arrows = observed topic transitions (thicker = more often).
"""
from __future__ import annotations

import textwrap
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")
from prepost_common import (COLORS, CONDITIONS, DARK, OUT, TIME_LABELS, counts_from, label_time, load_units_time,  # noqa: E402
                            pagerank, transition_matrix)

ORDER = ["VR Art", "VR Only", "Control"]  # as in topic_network_post figures
# each timepoint has its OWN topics (fit_timepoint_topics.py): topic ids of pre and post are unrelated
K_OF = {tm: [t for t in sorted(TIME_LABELS[tm]) if t >= 0] for tm in ("pre", "post")}
NODE_AREA = 1100.0  # marker area (pt^2) for a node with the maximal PageRank of the figure
MIN_COUNT = 1


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / name}.[pdf|png]")


def mds_layout(time):
    """MDS of the topic centroids of this timepoint's own descriptors (angular order on the rings)."""
    u = load_units_time(time)
    z = np.load(OUT / "unit_embeddings.npz")
    lk = dict(zip(z["document_id"].tolist(), z["emb"]))
    E = np.stack([lk[i] for i in u["document_id"]])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    K = K_OF[time]
    cent = np.stack([E[(u.Topic == t).to_numpy()].mean(axis=0) for t in K])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    dist = np.clip(1 - cent @ cent.T, 0, None)
    np.fill_diagonal(dist, 0)
    xy = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=8, normalized_stress="auto").fit_transform(dist)
    xy = xy - xy.mean(0)
    return {t: xy[i] for i, t in enumerate(K)}


def core_layout(pr, mds, time):
    K = K_OF[time]
    ranked = np.argsort(-pr)
    order = [K[i] for i in ranked]
    out = {order[0]: np.zeros(2)}
    for members, r, off in ((order[1:4], 1.9, 0.5), (order[4:], 3.7, 0.0)):
        members = sorted(members, key=lambda t: np.arctan2(mds[t][1], mds[t][0]))
        for i, t in enumerate(members):
            a = 2 * np.pi * (i + off) / len(members) + np.pi / 2
            out[t] = np.array([r * np.cos(a), r * np.sin(a)])
    return out


def panel_data(u, cond, time):
    K = K_OF[time]
    kidx = {t: i for i, t in enumerate(K)}
    sub = u[(u.condition_label == cond) & (u.time == time)]
    C = np.zeros((len(K), len(K)))
    for _, g in sub.groupby("participant_id"):
        arr = np.array([kidx[t] for t in g.Topic])
        C += counts_from(arr, np.zeros(len(arr), int), len(K))
    pr = pagerank(transition_matrix(C))
    return pr, C, sub.participant_id.nunique(), int(C.sum())


def draw_panel(ax, pr, C, mds, vmax, cmap, norm, title, color, time):
    K = K_OF[time]
    kidx = {t: i for i, t in enumerate(K)}
    xy = core_layout(pr, mds, time)
    ax.add_patch(plt.Circle((0, 0), 2.8, color=color, alpha=0.16, lw=0, zorder=0))
    rad = lambda p: (NODE_AREA * p / vmax / np.pi) ** 0.5  # noqa: E731
    for i, j in zip(*np.nonzero(C)):
        if C[i, j] < MIN_COUNT:
            continue
        ax.add_patch(FancyArrowPatch(xy[K[i]], xy[K[j]], connectionstyle="arc3,rad=0.16", arrowstyle="-|>", mutation_scale=9,
                                     lw=min(0.7 + 0.5 * C[i, j], 3.2), color="#555555", alpha=0.6,
                                     shrinkA=rad(pr[i]) + 2, shrinkB=rad(pr[j]) + 3, zorder=1))
    for t in K:
        p = pr[kidx[t]]
        ax.scatter(*xy[t], s=NODE_AREA * p / vmax, color=cmap(norm(p)), edgecolor="#333333", linewidth=1.0, zorder=3)
        txt = textwrap.fill(label_time(time, t), 13)
        box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
        if np.linalg.norm(xy[t]) > 0.5:
            u_ = xy[t] / np.linalg.norm(xy[t])
            ax.annotate(txt, xy[t], xytext=tuple(26 * u_), textcoords="offset points",
                        ha="left" if u_[0] > 0.3 else "right" if u_[0] < -0.3 else "center",
                        va="bottom" if u_[1] > 0.3 else "top" if u_[1] < -0.3 else "center", fontsize=10, zorder=5, bbox=box)
        else:
            ax.annotate(txt, xy[t], xytext=(0, 26), textcoords="offset points", ha="center", va="bottom", fontsize=10.5,
                        zorder=5, bbox=box)
    ax.set_title(title, fontsize=13, color=color, fontweight="bold", pad=8)
    ax.set_xlim(-6.4, 6.4)
    ax.set_ylim(-4.8, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    u = pd.concat([load_units_time("pre"), load_units_time("post")], ignore_index=True)
    mds = {tm: mds_layout(tm) for tm in ("pre", "post")}
    cmap = plt.get_cmap("magma_r")
    data = {(c, t): panel_data(u, c, t) for c in ORDER for t in ("pre", "post")}
    vmax = max(d[0].max() for d in data.values())
    norm = Normalize(0, vmax)
    sub_ = ("node size and colour = PageRank (same scale in all panels); arrows = observed transitions between consecutive "
            "descriptors (thicker = more often); most central topic in the middle")

    # ---- pre only
    fig, axes = plt.subplots(1, 3, figsize=(27, 8.6))
    for ax, c in zip(axes, ORDER):
        pr, C, n, nt = data[(c, "pre")]
        draw_panel(ax, pr, C, mds["pre"], vmax, cmap, norm, f"{c}  (pre; {n} participants, {nt} transitions)", DARK[c], "pre")
    cax = fig.add_axes([0.38, 0.06, 0.24, 0.02])
    fig.colorbar(ScalarMappable(norm, cmap), cax=cax, orientation="horizontal", label="PageRank")
    fig.suptitle("Core topic network of the pre-interview descriptors (topics extracted from the pre descriptors only), within response\n" + sub_, fontsize=12.5, y=0.985)
    fig.subplots_adjust(bottom=0.12, top=0.9, wspace=0.03)
    save(fig, "core_network_within_response_pre")

    # ---- pre vs post
    fig, axes = plt.subplots(2, 3, figsize=(27, 16.4))
    for r, time in enumerate(("pre", "post")):
        for ax, c in zip(axes[r], ORDER):
            pr, C, n, nt = data[(c, time)]
            draw_panel(ax, pr, C, mds[time], vmax, cmap, norm, f"{c} - {time}  ({n} participants, {nt} transitions)", DARK[c], time)
    cax = fig.add_axes([0.38, 0.045, 0.24, 0.012])
    fig.colorbar(ScalarMappable(norm, cmap), cax=cax, orientation="horizontal", label="PageRank")
    fig.suptitle("Core topic networks before (top) and after (bottom) VR, within response; each row uses the topics extracted from its own timepoint's descriptors\n" + sub_, fontsize=12.5, y=0.99)
    fig.subplots_adjust(bottom=0.08, top=0.93, wspace=0.03, hspace=0.06)
    save(fig, "core_network_pre_vs_post")


if __name__ == "__main__":
    main()
