#!/usr/bin/env python3
"""Figures for the topic-network centrality analysis.

For each graph (flow within responses, flow across questions, semantic kNN):
  * <name>_network.*     one network panel per condition, shared node layout
  * <name>_centrality.*  centrality per topic and condition, bootstrap 95% CI
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from sklearn.manifold import MDS

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from all_questions.topic_labels import short_topic_label, topic_label  # noqa: E402
from q3.plot_topic_differences import CONDITION_COLORS, CONDITION_ORDER  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs"
ASSIGNMENTS = (BASE_DIR / "all_questions" / "outputs"
               / "combined_questions_sentence_topic_assignments.csv")
GRAPHS = [
    ("flow", "within_response", "Flow graph: transitions within one response"),
    ("flow", "across_questions", "Flow graph: transitions across Q1-Q3 (per participant)"),
    ("semantic", "centroids", "Semantic graph: kNN on topic embedding centroids"),
]
MEASURE_TITLES = {
    "SBC_hop": "SBC (hop-shortest paths)",
    "PSBC_hop": "PSBC (hop-shortest, probability-weighted)",
    "SBC_prob": "SBC (most probable paths)",
    "SBC_sim": "SBC (most similar paths)",
}
NODE_MAX_AREA = 1400


def save(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {base}.[pdf|png]")


def layout(out: Path, topics: list[int]) -> dict[int, np.ndarray]:
    """2-D MDS of the topic centroids (cosine distance): semantically close = close."""
    cent = np.load(out / "topic_centroids_all.npy")
    dist = np.clip(1 - cent @ cent.T, 0, None)
    np.fill_diagonal(dist, 0)
    xy = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
             n_init=8, normalized_stress="auto").fit_transform(dist)
    xy = (xy - xy.mean(0)) / np.abs(xy - xy.mean(0)).max()
    return {t: xy[i] for i, t in enumerate(topics)}


def prevalence(topics: list[int]) -> pd.DataFrame:
    df = pd.read_csv(ASSIGNMENTS)
    df = df.loc[df["included_in_model"] & df["Topic"].isin(topics)]
    counts = df.groupby(["condition_label", "Topic"]).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).reindex(columns=topics, fill_value=0)


def draw_network(out, graph, scope, title, cent, edges, pos, prev, topics, labels):
    primary = "PSBC_hop"
    vmax = cent.loc[(cent.graph == graph) & (cent.scope == scope)
                    & (cent.measure == primary), "value"].max()
    norm = Normalize(0, max(vmax, 1e-6))
    cmap = plt.get_cmap("magma_r")
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.6))
    for ax, cond in zip(axes, CONDITION_ORDER):
        e = edges[(edges.graph == graph) & (edges.scope == scope) & (edges.condition == cond)]
        c = cent[(cent.graph == graph) & (cent.scope == scope) & (cent.measure == primary)
                 & (cent.condition == cond)].set_index("Topic")["value"]
        wmax = e["probability"].max() if len(e) else 1
        for r in e.itertuples():
            a, b = pos[r.source], pos[r.target]
            width = 0.4 + 4.0 * r.probability / wmax
            if graph == "flow":
                ax.add_patch(FancyArrowPatch(a, b, connectionstyle="arc3,rad=0.18",
                                             arrowstyle="-|>", mutation_scale=9,
                                             lw=width, color="#7a7a7a", alpha=0.45,
                                             shrinkA=9, shrinkB=9, zorder=1))
            else:
                ax.plot(*zip(a, b), lw=width, color="#7a7a7a", alpha=0.45, zorder=1)
        for t in topics:
            p = prev.loc[cond, t]
            if p <= 0:
                continue
            ax.scatter(*pos[t], s=NODE_MAX_AREA * p / prev.values.max(),
                       color=cmap(norm(c.get(t, 0))), edgecolor="#333333",
                       linewidth=0.8, zorder=3)
            ax.text(*pos[t], str(t), ha="center", va="center", fontsize=9,
                    fontweight="bold", zorder=4,
                    color="white" if norm(c.get(t, 0)) > 0.55 else "#111111")
        ax.set_title(cond, fontsize=15, color=CONDITION_COLORS[cond], fontweight="bold")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
    cax = fig.add_axes([0.35, 0.16, 0.3, 0.02])
    fig.colorbar(ScalarMappable(norm, cmap), cax=cax, orientation="horizontal",
                 label=f"{MEASURE_TITLES[primary]}  (node colour)")
    items = [f"{t}: {labels[t]}" for t in topics]
    per = -(-len(items) // 3)
    key = "\n".join("     ".join(items[i:i + per]) for i in range(0, len(items), per))
    fig.text(0.5, 0.085, key, ha="center", va="top", fontsize=9,
             color="#444444")
    fig.suptitle(f"{title}\nnode size = topic share in condition; layout = embedding MDS "
                 f"(shared across panels)", fontsize=13, y=1.0)
    fig.subplots_adjust(bottom=0.24, top=0.88, wspace=0.02)
    save(fig, out / f"network_{graph}_{scope}")


def draw_centrality(out, graph, scope, title, cent, topics, labels):
    sub = cent[(cent.graph == graph) & (cent.scope == scope)]
    measures = [m for m in ["SBC_hop", "PSBC_hop", "SBC_prob", "SBC_sim"]
                if m in set(sub.measure)]
    order = (sub[sub.measure == "PSBC_hop"].groupby("Topic")["value"].mean()
             .sort_values(ascending=False).index.tolist())
    ypos = {t: len(order) - 1 - i for i, t in enumerate(order)}
    offsets = {"VR Art": 0.24, "VR Only": 0.0, "Control": -0.24}
    fig, axes = plt.subplots(1, len(measures), figsize=(5.6 * len(measures) + 3.2, 7),
                             sharey=True)
    for ax, m in zip(axes, measures):
        for cond in CONDITION_ORDER:
            d = sub[(sub.measure == m) & (sub.condition == cond)].set_index("Topic")
            y = [ypos[t] + offsets[cond] for t in order]
            v = d.loc[order, "value"].to_numpy()
            lo = d.loc[order, "ci_low"].to_numpy()
            hi = d.loc[order, "ci_high"].to_numpy()
            ax.errorbar(v, y, xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)],
                        fmt="o", ms=6, mec="white", mew=0.7, color=CONDITION_COLORS[cond],
                        ecolor=CONDITION_COLORS[cond], elinewidth=1.4, capsize=2,
                        label=cond, zorder=3)
        ax.set_title(MEASURE_TITLES[m], fontsize=11)
        ax.set_xlim(left=-0.005)
        ax.set_xlabel("Centrality (normalised)")
        ax.grid(axis="x", color="#dddddd")
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[t] for t in order])
    axes[0].set_yticklabels([labels[t] for t in order], fontsize=10.5)
    axes[0].legend(loc="lower center", bbox_to_anchor=(len(measures) / 2, 1.09),
                   ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"{title}  (points: observed, bars: participant-bootstrap 95% CI)",
                 fontsize=12, y=1.10)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out / f"centrality_{graph}_{scope}")


CENTRAL_PANELS = [
    ("flow", "within_response", "PageRank", "PageRank\n(flow within responses)"),
    ("flow", "across_questions", "PageRank", "PageRank\n(flow across Q1-Q3)"),
    ("semantic", "centroids", "SemCentrality", "Semantic centrality\n(similarity to rest of condition)"),
]


def draw_central(out, cent, topics, labels):
    """What is central to each condition: PageRank on the flow graphs + semantic centrality."""
    def panel(g, sc, m):
        return cent[(cent.graph == g) & (cent.scope == sc) & (cent.measure == m)]

    order = (panel(*CENTRAL_PANELS[0][:3]).groupby("Topic")["value"].mean()
             .sort_values(ascending=False).index.tolist())
    ypos = {t: len(order) - 1 - i for i, t in enumerate(order)}
    offsets = {"VR Art": 0.24, "VR Only": 0.0, "Control": -0.24}
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
    for ax, (g, sc, m, title) in zip(axes, CENTRAL_PANELS):
        sub = panel(g, sc, m)
        for cond in CONDITION_ORDER:
            d = sub[sub.condition == cond].set_index("Topic")
            v = d.loc[order, "value"].to_numpy()
            lo = d.loc[order, "ci_low"].to_numpy()
            hi = d.loc[order, "ci_high"].to_numpy()
            ax.errorbar(v, [ypos[t] + offsets[cond] for t in order],
                        xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)],
                        fmt="o", ms=6.5, mec="white", mew=0.7, color=CONDITION_COLORS[cond],
                        ecolor=CONDITION_COLORS[cond], elinewidth=1.4, capsize=2,
                        label=cond, zorder=3)
        if m == "PageRank":
            ax.axvline(1 / len(topics), color="#888888", ls="--", lw=1, zorder=1)
            ax.text(1 / len(topics), len(order) - 0.35, " uniform", fontsize=8,
                    color="#888888", va="bottom")
            ax.set_xlim(left=0)
        ax.set_title(title, fontsize=11.5)
        ax.grid(axis="x", color="#dddddd")
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[t] for t in order])
    axes[0].set_yticklabels([labels[t] for t in order], fontsize=10.5)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.65, 1.13), ncol=3,
                   frameon=False, fontsize=12)
    fig.suptitle("What is central to each condition  (points: observed, bars: "
                 "participant-bootstrap 95% CI)", fontsize=12, y=1.10)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out / "central_topics_by_condition")


def core_layout(pagerank: pd.Series, pos: dict, topics: list[int]) -> dict:
    """Radial layout: top PageRank topic in the middle, next 3 on an inner ring,
    the rest on an outer ring; angular order follows the shared embedding MDS."""
    ranked = pagerank.sort_values(ascending=False).index.tolist()
    rings = [(ranked[:1], 0.0), (ranked[1:4], 1.9), (ranked[4:], 3.7)]
    out = {}
    for members, r in rings:
        if r == 0:
            out[members[0]] = np.zeros(2)
            continue
        members = sorted(members, key=lambda t: np.arctan2(pos[t][1], pos[t][0]))
        offset = 0.5 if r < 2 else 0.0
        for i, t in enumerate(members):
            a = 2 * np.pi * (i + offset) / len(members) + np.pi / 2
            out[t] = np.array([r * np.cos(a), r * np.sin(a)])
    return out


def draw_core_network(out, scope, title, cent, edges, pos, prev, topics, labels, min_count=2):
    pr = cent[(cent.graph == "flow") & (cent.scope == scope) & (cent.measure == "PageRank")]
    norm = Normalize(0, pr["value"].max())
    cmap = plt.get_cmap("magma_r")
    fig, axes = plt.subplots(1, 3, figsize=(28, 9.5))
    for ax, cond in zip(axes, CONDITION_ORDER):
        c = pr[pr.condition == cond].set_index("Topic")["value"]
        xy = core_layout(c, pos, topics)
        ax.add_patch(plt.Circle((0, 0), 2.8, color=CONDITION_COLORS[cond], alpha=0.16,
                                lw=0, zorder=0))
        e = edges[(edges.graph == "flow") & (edges.scope == scope)
                  & (edges.condition == cond) & (edges["count"] >= min_count)]
        for r in e.itertuples():
            rs = (NODE_MAX_AREA * 0.9 * c[r.source] / pr["value"].max() / np.pi) ** 0.5
            rt = (NODE_MAX_AREA * 0.9 * c[r.target] / pr["value"].max() / np.pi) ** 0.5
            ax.add_patch(FancyArrowPatch(
                xy[r.source], xy[r.target], connectionstyle="arc3,rad=0.16",
                arrowstyle="-|>", mutation_scale=9, lw=min(0.7 + 0.5 * r.count, 3.2),
                color="#555555", alpha=0.6, shrinkA=rs + 2, shrinkB=rt + 3, zorder=1))
        for t in topics:
            ax.scatter(*xy[t], s=NODE_MAX_AREA * 0.9 * c[t] / pr["value"].max(),
                       color=cmap(norm(c[t])), edgecolor="#333333", linewidth=1.0, zorder=3)
            txt = textwrap.fill(labels[t], 20)
            box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
            if np.linalg.norm(xy[t]) > 0.5:  # rings: label radially outward
                u = xy[t] / np.linalg.norm(xy[t])
                ax.annotate(txt, xy[t], xytext=tuple(30 * u), textcoords="offset points",
                            ha="left" if u[0] > 0.3 else "right" if u[0] < -0.3 else "center",
                            va="bottom" if u[1] > 0.3 else "top" if u[1] < -0.3 else "center",
                            fontsize=11.5, zorder=5, bbox=box)
            else:  # centre: narrow label above the node
                ax.annotate(textwrap.fill(labels[t], 12), xy[t], xytext=(0, 28), textcoords="offset points",
                            ha="center", va="bottom", fontsize=11.5, zorder=5, bbox=box)
        ax.set_title(cond, fontsize=16, color=CONDITION_COLORS[cond], fontweight="bold", pad=14)
        ax.set_xlim(-6.6, 6.6)
        ax.set_ylim(-4.6, 5.0)
        ax.set_aspect("equal")
        ax.axis("off")
    cax = fig.add_axes([0.38, 0.06, 0.24, 0.02])
    fig.colorbar(ScalarMappable(norm, cmap), cax=cax, orientation="horizontal",
                 label="PageRank (node size and colour)")
    fig.suptitle(f"{title}\nmost central topic in the middle, then inner and outer ring; "
                 f"arrows: transitions seen at least {min_count} times, thicker = more",
                 fontsize=13, y=0.99)
    fig.subplots_adjust(bottom=0.1, top=0.9, wspace=0.04)
    save(fig, out / f"core_network_{scope}")


SCOPE_TITLES = {"within_response": "flow within responses",
                "across_questions": "flow across Q1-Q3"}


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def draw_pagerank_vs_null(out, scope, tests, labels):
    """Observed PageRank vs the shuffled-order null, per condition."""
    sub = tests[tests.scope == scope]
    order = (sub.groupby("Topic")["pagerank"].mean().sort_values(ascending=False).index.tolist())
    ypos = {t: len(order) - 1 - i for i, t in enumerate(order)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8), sharey=True, sharex=True)
    for ax, cond in zip(axes, CONDITION_ORDER):
        d = sub[sub.condition == cond].set_index("Topic")
        for t in order:
            r, y = d.loc[t], ypos[t]
            ax.plot([r.null_low, r.null_high], [y, y], lw=9, color="#d9d9d9",
                    solid_capstyle="round", zorder=1)
            ax.plot(r.null_mean, y, "|", color="#777777", ms=11, mew=2, zorder=2)
            ax.errorbar(r.pagerank, y, xerr=[[max(r.pagerank - r.ci_low, 0)],
                                             [max(r.ci_high - r.pagerank, 0)]],
                        fmt="o", ms=8 if r.is_top else 6.5, mec="#333333" if r.is_top else "white",
                        mew=1.2 if r.is_top else 0.7, color=CONDITION_COLORS[cond],
                        ecolor=CONDITION_COLORS[cond], elinewidth=1.4, capsize=2, zorder=3)
            if stars(r.p_fdr):
                ax.text(max(r.ci_high, r.null_high) + 0.006, y, stars(r.p_fdr), va="center",
                        fontsize=13, fontweight="bold")
        ax.set_title(cond, fontsize=14, color=CONDITION_COLORS[cond], fontweight="bold")
        ax.set_xlim(left=0)
        ax.set_xlabel("PageRank")
        ax.grid(axis="x", color="#e2e2e2")
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[t] for t in order])
    axes[0].set_yticklabels([labels[t] for t in order], fontsize=11.5)
    fig.suptitle(f"PageRank vs. chance ({SCOPE_TITLES[scope]})\n"
                 "dot = observed with participant-bootstrap 95% CI (outlined = most central); "
                 "grey bar = 95% range if sentence order were random (| = mean); "
                 "* = FDR-adjusted p < .05, ** < .01, *** < .001", fontsize=11, y=1.02)
    fig.subplots_adjust(wspace=0.05)
    save(fig, out / f"pagerank_vs_null_{scope}")


def draw_contributions(out, scope, contrib, labels):
    """Who feeds the most central topic of each condition."""
    sub = contrib[contrib.scope == scope]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    for ax, cond in zip(axes, CONDITION_ORDER):
        d = sub[sub.condition == cond].copy()
        d["short"] = [("Baseline\n(teleport + dangling)" if t == -99 else labels[t])
                      for t in d.source_topic]
        d = d[(d.contribution > 1e-4) | (d.source_topic == -99)]
        d = d.sort_values("contribution", ascending=True)
        y = np.arange(len(d))
        for yi, r in zip(y, d.itertuples()):
            base = r.source_topic == -99
            sig = (not base) and r.p_fdr < 0.05
            ax.barh(yi, r.contribution, color="#b9b9b9" if base else CONDITION_COLORS[cond],
                    alpha=1.0 if sig else 0.45, edgecolor="#333333" if sig else "none",
                    lw=1.0, zorder=2)
            if not base and pd.notna(r.ci_low):
                ax.plot([r.ci_low, r.ci_high], [yi, yi], color="#444444", lw=1.2, zorder=3)
                ax.plot(r.null_mean, yi, "|", color="#222222", ms=13, mew=2, zorder=4)
                txt = f"{stars(r.p_fdr)}" if sig else ("" if pd.isna(r.p_fdr) else "")
                ax.text(max(r.contribution, r.ci_high) + 0.003, yi,
                        f"{int(r.n_transitions_to_target)}x {txt}", va="center", fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels(d["short"], fontsize=11)
        top = labels[int(d["target_topic"].iloc[0])]
        ax.set_title(f"{cond}\nmost central: {top} (PR {d.target_pagerank.iloc[0]:.2f})",
                     fontsize=12, color=CONDITION_COLORS[cond], fontweight="bold")
        ax.set_xlabel("Contribution to its PageRank")
        ax.grid(axis="x", color="#e2e2e2")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"What feeds the most central topic ({SCOPE_TITLES[scope]})\n"
                 "bar = contribution a*PR(source)*P(source->target); line = bootstrap 95% CI; "
                 "| = expected if sentence order were random; label = observed transitions "
                 "into the target; solid bar + * = FDR p < .05 (one-sided)", fontsize=11, y=1.03)
    fig.subplots_adjust(wspace=0.75)
    save(fig, out / f"pagerank_contributions_{scope}")


def draw_path_distributions(out, scope, reach, routes, tests, labels, smooth=0.1):
    """Paths toward each central topic, compared across conditions."""
    reach = reach[(reach.scope == scope) & (reach.smooth == smooth)]
    routes = routes[(routes.scope == scope) & (routes.smooth == smooth)]
    tests = tests[(tests.scope == scope) & (tests.smooth == smooth)]
    targets = list(dict.fromkeys(reach["target_topic"]))
    offsets = {"VR Art": 0.24, "VR Only": 0.0, "Control": -0.24}
    fig, axes = plt.subplots(len(targets), 2, figsize=(15, 5.2 * len(targets)),
                             gridspec_kw={"width_ratios": [1, 2.6]}, squeeze=False)
    for row, t in enumerate(targets):
        axr, axe = axes[row]
        rr = reach[reach.target_topic == t]
        for i, cond in enumerate(CONDITION_ORDER):
            d = rr[rr.condition == cond].iloc[0]
            axr.errorbar(i, d.reach_prob, yerr=[[d.reach_prob - d.ci_low], [d.ci_high - d.reach_prob]],
                         fmt="o", ms=9, mec="white", color=CONDITION_COLORS[cond],
                         ecolor=CONDITION_COLORS[cond], elinewidth=1.8, capsize=4)
        axr.set_xticks(range(3))
        axr.set_xticklabels(CONDITION_ORDER, fontsize=11)
        axr.set_xlim(-0.6, 2.6)
        axr.set_ylim(0, 1)
        axr.set_ylabel("Probability of reaching the target\n(within 3 steps)")
        tt = tests[(tests.target_topic == t) & (tests.test == "reach_probability")]
        axr.set_title("reach:  " + "  ".join(
            f"{a[:5]}-{b[:5]} p={p:.2f}" for a, b, p in zip(tt.condition_a.str.replace(" ", ""),
                                                          tt.condition_b.str.replace(" ", ""), tt.p_perm)),
            fontsize=8.5)
        target_label = labels[t]
        er = routes[routes.target_topic == t]
        order = er.groupby("source_topic")["entry_mass"].mean().sort_values(ascending=False).index.tolist()
        ypos = {s_: len(order) - 1 - i for i, s_ in enumerate(order)}
        et = tests[(tests.target_topic == t) & (tests.test == "entry_mass")]
        for cond in CONDITION_ORDER:
            d = er[er.condition == cond].set_index("source_topic")
            v = d.loc[order, "entry_mass"].to_numpy()
            lo = d.loc[order, "ci_low"].to_numpy()
            hi = d.loc[order, "ci_high"].to_numpy()
            axe.errorbar(v, [ypos[s_] + offsets[cond] for s_ in order],
                         xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)],
                         fmt="o", ms=6.5, mec="white", mew=0.7, color=CONDITION_COLORS[cond],
                         ecolor=CONDITION_COLORS[cond], elinewidth=1.3, capsize=2, label=cond, zorder=3)
        for s_ in order:
            p = et[et.source_label == topic_label(s_)]["p_perm"].min()
            if p < 0.05:
                axe.text(er["ci_high"].max() * 1.03, ypos[s_], "†", fontsize=13, va="center",
                         fontweight="bold")
        axe.set_yticks([ypos[s_] for s_ in order])
        axe.set_yticklabels([labels[s_] for s_ in order], fontsize=11)
        axe.set_xlim(left=-0.005)
        axe.set_xlabel(f"Probability of entering '{target_label}' via this topic (last stop before it)")
        jt = tests[(tests.target_topic == t) & (tests.test == "route_distribution_jsd")]
        axe.set_title("route distribution (JSD):  " + "  ".join(
            f"{a[:5]}-{b[:5]} p={p:.2f}" for a, b, p in zip(jt.condition_a.str.replace(" ", ""),
                                                          jt.condition_b.str.replace(" ", ""), jt.p_perm)),
            fontsize=8.5)
        axr.text(-0.16, 1.16, f"Target: {target_label}", transform=axr.transAxes, fontsize=14,
                 fontweight="bold")
        axe.grid(axis="x", color="#e2e2e2")
        axe.set_axisbelow(True)
        axr.grid(axis="y", color="#e2e2e2")
        axr.set_axisbelow(True)
        for a in (axr, axe):
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        if row == 0:
            axe.legend(loc="lower right", frameon=False, fontsize=11)
    fig.suptitle(f"Paths toward each central topic ({SCOPE_TITLES[scope]}): first arrival within "
                 "3 steps from the opening topic; bars = participant-bootstrap 95% CI; "
                 "† = uncorrected permutation p < .05 for a between-condition difference "
                 "(none survive FDR)", fontsize=10.5, y=0.995)
    fig.subplots_adjust(hspace=0.62, wspace=0.42, top=0.9)
    save(fig, out / f"path_distribution_{scope}")


def draw_global_tests(out, contrib, labels):
    """Omnibus permutation tests + which topics drive the divergence."""
    z = np.load(out / "global_distribution_null.npz")
    panels = [("composition_gjsd", "Topic composition\n(pooled GJSD, bits)"),
              ("composition_permanova_F", "Topic composition\n(participant PERMANOVA F)"),
              ("transition_gjsd__within_response", "Transitions within responses\n(source-weighted GJSD)"),
              ("transition_gjsd__across_questions", "Transitions across Q1-Q3\n(source-weighted GJSD)"),
              ("stationary_gjsd__within_response", "Stationary distribution, within\n(GJSD of PageRank)"),
              ("stationary_gjsd__across_questions", "Stationary distribution, across\n(GJSD of PageRank)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, title) in zip(axes.ravel(), panels):
        nl, ob = z[key], float(z["obs__" + key])
        p = (1 + np.sum(nl >= ob - 1e-12)) / (1 + len(nl))
        ax.hist(nl, bins=40, color="#cfcfcf", edgecolor="white")
        ax.axvline(ob, color="#c0392b", lw=2.2)
        ax.set_title(f"{title}\nobserved {ob:.3f}, permutation p = {p:.3f}", fontsize=10.5)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Are topics distributed differently across conditions?  "
                 "grey = null (participants shuffled across conditions), red = observed",
                 fontsize=12, y=1.0)
    fig.subplots_adjust(hspace=0.65, wspace=0.15)
    save(fig, out / "global_distribution_tests")

    specs = [("topic (composition)", "all", "Topic composition\n(contribution to GJSD)"),
             ("source topic (transitions)", "within_response", "Transitions within responses\n(contribution by source topic)"),
             ("source topic (transitions)", "across_questions", "Transitions across Q1-Q3\n(contribution by source topic)")]
    order = (contrib[contrib.family == "topic (composition)"].sort_values("contribution", ascending=False)
             ["Topic"].tolist())
    ypos = {t: len(order) - 1 - i for i, t in enumerate(order)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True)
    for ax, (fam, scope, title) in zip(axes, specs):
        d = contrib[(contrib.family == fam) & (contrib.scope == scope)].set_index("Topic")
        for t in order:
            r, y = d.loc[t], ypos[t]
            ax.plot([r.null_low, r.null_high], [y, y], lw=9, color="#d9d9d9", solid_capstyle="round", zorder=1)
            ax.plot(r.null_mean, y, "|", color="#777777", ms=11, mew=2, zorder=2)
            ax.plot(r.contribution, y, "o", ms=7, color="#c0392b" if r.p_fdr < 0.05 else "#555555",
                    mec="white", zorder=3)
            if r.p_fdr < 0.05:
                ax.text(max(r.contribution, r.null_high) * 1.05, y, "*", fontsize=14, va="center",
                        fontweight="bold")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(left=0)
        ax.grid(axis="x", color="#e2e2e2")
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[t] for t in order])
    axes[0].set_yticklabels([labels[t] for t in order], fontsize=11)
    fig.suptitle("Which topics drive the divergence?  dot = observed, grey bar = 95% range under "
                 "shuffled condition labels (| = mean), * = FDR p < .05", fontsize=11, y=1.0)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out / "global_distribution_contributions")


def draw_path_graph_jsd(out, tests):
    """JSD between the path distributions implied by the condition graphs."""
    z = np.load(out / "path_graph_jsd_null.npz")
    main = tests[(tests.path_length == 3) & (tests.start == "pooled") & (tests.smooth == 0.1)]
    cmp_names = ["all three", "VR Art vs VR Only", "VR Art vs Control", "VR Only vs Control"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.6))
    for r, scope in enumerate(("within_response", "across_questions")):
        nl, ob = z[scope], z["obs_" + scope]
        for c, name in enumerate(cmp_names):
            ax = axes[r, c]
            ax.hist(nl[:, c], bins=40, color="#cfcfcf", edgecolor="white")
            ax.axvline(ob[c], color="#c0392b", lw=2.2)
            p = main[(main.scope == scope) & (main.comparison == name)]["p_perm"].iloc[0]
            ax.set_title(f"{SCOPE_TITLES[scope]}\n{name}\nJSD {ob[c]:.3f} bits, p = {p:.3f}", fontsize=9.5)
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("JSD between the path distributions of the condition graphs (3-step paths, "
                 "shared opening-topic distribution)  grey = participants shuffled across conditions, "
                 "red = observed", fontsize=11, y=1.0)
    fig.subplots_adjust(hspace=0.85, wspace=0.12)
    save(fig, out / "path_graph_jsd_tests")

    top = pd.read_csv(out / "path_graph_jsd_top_paths.csv")
    full2short = {}
    for t in range(-1, 12):
        full2short[topic_label(t)] = short_topic_label(t)
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.2))
    for ax, scope in zip(axes, ("within_response", "across_questions")):
        d = top[top.scope == scope].head(8).iloc[::-1].reset_index(drop=True)
        y = np.arange(len(d))
        for k, cond in enumerate(CONDITION_ORDER):
            ax.barh(y + (1 - k) * 0.26, d[f"prob_{cond}"], height=0.25, color=CONDITION_COLORS[cond],
                    label=cond)
        ax.set_yticks(y)
        ax.set_yticklabels([" > ".join(full2short.get(x, x) for x in p.split(" > ")) for p in d["path"]],
                           fontsize=9.5)
        ax.set_xlabel("Probability of the path (under the condition's graph)")
        ax.set_title(SCOPE_TITLES[scope], fontsize=12)
        ax.grid(axis="x", color="#e2e2e2")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=11, loc="lower right")
    fig.suptitle("Paths that contribute most to the divergence between the three condition graphs",
                 fontsize=12, y=1.0)
    fig.subplots_adjust(wspace=1.45)
    save(fig, out / "path_graph_jsd_top_paths")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    out = p.parse_args().output_dir.expanduser().resolve()
    cent = pd.read_csv(out / "centrality_by_condition.csv")
    edges = pd.read_csv(out / "network_edges.csv")
    topics = sorted(cent["Topic"].unique().tolist())
    labels = {t: short_topic_label(t) for t in topics}
    draw_central(out, cent, topics, labels)
    tests = pd.read_csv(out / "pagerank_within_condition_topic_tests.csv")
    contrib = pd.read_csv(out / "pagerank_top_topic_contributions.csv")
    draw_path_graph_jsd(out, pd.read_csv(out / "path_graph_jsd_tests.csv"))
    draw_global_tests(out, pd.read_csv(out / "global_distribution_contributions.csv"), labels)
    reach = pd.read_csv(out / "path_reach_by_condition.csv")
    routes = pd.read_csv(out / "path_entry_routes_by_condition.csv")
    ptests = pd.read_csv(out / "path_condition_tests.csv")
    for scope in ("within_response", "across_questions"):
        draw_path_distributions(out, scope, reach, routes, ptests, labels)
        draw_pagerank_vs_null(out, scope, tests, labels)
        draw_contributions(out, scope, contrib, labels)
    pos = layout(out, topics)
    prev = prevalence(topics)
    for graph, scope, title in GRAPHS:
        if graph == "flow":
            draw_core_network(out, scope, title, cent, edges, pos, prev, topics, labels)
        draw_network(out, graph, scope, title, cent, edges, pos, prev, topics, labels)
        draw_centrality(out, graph, scope, title, cent, topics, labels)


if __name__ == "__main__":
    main()
