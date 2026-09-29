#!/usr/bin/env python3
"""Figures for the pre/post analysis.

  prepost_topic_shares.*     topic share pre -> post per condition (paired participants)
  prepost_semantic_space.*   semantic-space map with pre -> post shifts, and per-PC change
  prepost_pagerank.*         PageRank pre vs post per condition (participant bootstrap CIs)
  prepost_tests_overview.*   all test p-values in one table
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
from prepost_common import (COLORS, CONDITIONS, DARK, OUT, TOPIC_LABELS, embeddings, label, load_units,  # noqa: E402
                            participants)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / name}.[pdf|png]")


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def topic_shares():
    tc = pd.read_csv(OUT / "prepost_topic_changes.csv")
    tc = tc[tc.condition != "interaction (ANOVA F)"]
    order = tc[tc.condition == "all conditions"].sort_values("change").topic.tolist()  # most negative first
    ypos = {t: len(order) - 1 - i for i, t in enumerate(order)}
    panels = ["all conditions"] + CONDITIONS
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.2), sharey=True, sharex=True)
    for ax, c in zip(axes, panels):
        d = tc[tc.condition == c].set_index("topic")
        col = "#555555" if c == "all conditions" else COLORS[c]
        dark = "#333333" if c == "all conditions" else DARK[c]
        for t in order:
            r = d.loc[t]
            y = ypos[t]
            ax.add_patch(FancyArrowPatch((100 * r.share_pre, y), (100 * r.share_post, y), arrowstyle="-|>",
                                         mutation_scale=11, lw=2, color=col, zorder=2, shrinkA=4, shrinkB=4))
            ax.scatter(100 * r.share_pre, y, s=48, facecolor="white", edgecolor=dark, lw=1.6, zorder=3)
            ax.scatter(100 * r.share_post, y, s=52, color=dark, zorder=4)
            if stars(r.p_fdr):
                ax.text(max(r.share_pre, r.share_post) * 100 + 1.8, y, stars(r.p_fdr), va="center",
                        fontsize=13, fontweight="bold")
        n = int(d.n.iloc[0])
        ax.set_title(f"{c.replace('all conditions', 'All conditions')}  (n = {n})", fontsize=12.5,
                     color="#222222" if c == "all conditions" else DARK[c], fontweight="bold")
        ax.set_xlabel("Share of participant's descriptors (%)")
        ax.grid(axis="x", color="#e6e6e6")
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[t] for t in order])
    axes[0].set_yticklabels(order, fontsize=11)
    fig.suptitle("Topic shares before (open) and after (filled) VR; paired participants; "
                 "* = BH-adjusted p < .05, ** < .01 (sign-flip permutation within participants)", fontsize=11.5, y=1.0)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "prepost_topic_shares")


PC_END_LABELS = {  # interpretation from topic positions on the PC (see README)
    1: ("Calm & fine", "Worries"),
    3: ("Anticipation", "Calm & video"),
}
SEM_COLS = [("all conditions", "All"), ("VR Art", "VR Art"), ("VR Only", "VR Only"), ("Control", "Control")]


def semantic_space():
    u = load_units()
    E = embeddings(u)
    P = participants(u)
    pca = PCA(n_components=10, random_state=0).fit(E)
    S = pca.transform(E)
    sd = S.std(axis=0)
    ev = pca.explained_variance_ratio_
    both = P[P.has_pre & P.has_post].reset_index(drop=True)

    def cent(pid, t):
        return S[((u.participant_id == pid) & (u.time == t)).to_numpy()].mean(axis=0)

    pre = np.stack([cent(i, "pre") for i in both.participant_id])
    post = np.stack([cent(i, "post") for i in both.participant_id])
    lab = both.cond.to_numpy()
    tests = pd.read_csv(OUT / "prepost_content_tests.csv")
    pcs = pd.read_csv(OUT / "prepost_pc_changes.csv")
    rng = np.random.default_rng(1)
    # axes = the two PCs that separate pre from post best (smallest FDR p, pooled), signs oriented so that
    # the mean change post - pre is positive on both axes
    pooled = pcs[pcs.condition == "all conditions"].sort_values("p_perm")
    AX = (pooled[pooled.p_fdr < 0.05].PC.astype(int).head(2) - 1).tolist()
    sign = np.sign((post - pre).mean(axis=0))
    flip = {k: sign[k] for k in AX}
    S = S * sign  # flips every PC so that the mean change is positive; only the two chosen axes are shown
    pre, post = pre * sign, post * sign

    def p_of(analysis, test, comp):
        r = tests[(tests.analysis == analysis) & (tests.test.str.startswith(test)) & (tests.comparison == comp)]
        return float(r.p_perm.iloc[0])

    def fdr3(test):
        ps = np.array([p_of("pre vs post (paired)", test, c) for c in CONDITIONS])
        from prepost_common import bh_fdr
        return dict(zip(CONDITIONS, bh_fdr(ps)))

    fdr_sem, fdr_top = fdr3("semantic"), fdr3("topic")
    within = pcs[pcs.condition.isin(CONDITIONS)]

    fig = plt.figure(figsize=(13, 9.6))
    ax = fig.add_axes([0.065, 0.06, 0.925, 0.86])
    for c, cname in enumerate(CONDITIONS):
        m = lab == c
        for a_, b_ in zip(pre[m][:, AX], post[m][:, AX]):
            ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=COLORS[cname], lw=0.9, alpha=0.5, zorder=1)
        ax.scatter(pre[m][:, AX[0]], pre[m][:, AX[1]], s=20, facecolor="white", edgecolor=COLORS[cname], lw=1, zorder=2)
        ax.scatter(post[m][:, AX[0]], post[m][:, AX[1]], s=24, color=COLORS[cname], zorder=2)
    for c, cname in enumerate(CONDITIONS):
        m = lab == c
        a_, b_ = pre[m].mean(axis=0)[AX], post[m].mean(axis=0)[AX]
        ax.add_patch(FancyArrowPatch(a_, b_, arrowstyle="-|>", mutation_scale=22, lw=3.2, color=DARK[cname],
                                     zorder=5, shrinkA=8, shrinkB=8))
        ax.scatter(*a_, s=170, marker="D", facecolor="white", edgecolor=DARK[cname], lw=2.2, zorder=6)
        ax.scatter(*b_, s=170, marker="D", color=DARK[cname], edgecolor="white", lw=1.4, zorder=7,
                   label=cname)
        sig_pcs = within[(within.condition == cname) & (within.p_fdr < 0.05)].PC.astype(int).tolist()
        if fdr_sem[cname] < 0.05 or sig_pcs:
            txt = "*" + ("  " + ", ".join(f"PC{k}" for k in sig_pcs) if sig_pcs else "")
            ax.annotate(txt, b_, xytext=(14, 2), textcoords="offset points", fontsize=17, fontweight="bold",
                        color=DARK[cname], zorder=8, va="center")
    def axname(k):
        return f"PC{k + 1} ({100 * ev[k]:.1f}% of variance" + ("; sign flipped" if flip[k] < 0 else "") + ")"

    ax.set_xlabel(axname(AX[0]))
    ax.set_ylabel(axname(AX[1]))
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=10.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#eeeeee")
    ax.set_axisbelow(True)
    ax.set_title("Semantic space of Q1 descriptors, pre -> post (paired participants)\n"
                 "thin lines = participants, open -> filled diamonds = condition means (pre -> post); axes = the two PCs that move most (picked for display); baseline (pre) conditions do not differ: "
                 "composition p = {:.2f}, semantic space p = {:.2f}\n"
                 "pre -> post shift, all conditions: p = {:.4f}".format(
                     p_of("baseline (pre only)", "topic", "all three"), p_of("baseline (pre only)", "semantic", "all three"),
                     p_of("pre vs post (paired)", "semantic", "all conditions")), fontsize=10.5)
    ys = np.concatenate([pre[:, AX[1]], post[:, AX[1]]])
    yr = ys.max() - ys.min()
    ax.set_ylim(ys.min() - 0.30 * yr, ys.max() + 0.03 * yr)

    # ---- inset 1: heat table of pre -> post and interaction p-values (no stars)
    box = ax.inset_axes([0.012, 0.018, 0.375, 0.165])
    box.set_zorder(10)
    box.set_xticks([])
    box.set_yticks([])
    box.set_facecolor("white")
    for sp in box.spines.values():
        sp.set_edgecolor("#555555")
        sp.set_linewidth(1.3)
    axt = box.inset_axes([0.335, 0.03, 0.655, 0.62])
    cmap = LinearSegmentedColormap.from_list("p", ["#c0392b", "#f5b7a5", "#ffffff"])
    rows = [("Topic composition\n(shared topics)", "topic"), ("Semantic space", "semantic")]
    cols = SEM_COLS + [("interaction", "Change\ndiffers?")]
    for i, (rname, key) in enumerate(rows):
        for j, (cn, _) in enumerate(cols):
            if cn == "interaction":
                v = p_of("interaction (change differs by condition)", key, "all three")
            elif cn == "all conditions":
                v = p_of("pre vs post (paired)", key, cn)
            else:
                v = (fdr_sem if key == "semantic" else fdr_top)[cn]  # BH-FDR over the three conditions
            y = len(rows) - 1 - i
            axt.add_patch(plt.Rectangle((j, y), 1, 1, color=cmap(np.clip(v / 0.2, 0, 1)), ec="white", lw=2))
            axt.text(j + 0.5, y + 0.5, "<.001" if v < 0.001 else f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                     fontweight="bold" if v < 0.05 else "normal")
    axt.set_xlim(0, len(cols))
    axt.set_ylim(0, len(rows))
    axt.set_xticks(np.arange(len(cols)) + 0.5)
    axt.set_xticklabels([l for _, l in cols], fontsize=8)
    axt.xaxis.tick_top()
    axt.set_yticks(np.arange(len(rows)) + 0.5)
    axt.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    axt.tick_params(length=0)
    for sp in axt.spines.values():
        sp.set_visible(False)
    box.text(0.5, 0.925, "Pre -> post shift (FDR-adjusted p over conditions)", ha="center", va="center", fontsize=8.2,
             fontweight="bold")

    # ---- inset 2: PCs that move pre -> post, with group changes and end labels (no stars)
    allr = pcs[pcs.condition == "all conditions"]
    keep_pcs = sorted(set(allr[allr.p_fdr < 0.05].PC) | set(within[within.p_fdr < 0.05].PC))
    sig = allr[allr.PC.isin(keep_pcs)].reset_index(drop=True)
    box2 = ax.inset_axes([0.398, 0.018, 0.375, 0.165])
    box2.set_zorder(10)
    box2.set_xticks([])
    box2.set_yticks([])
    box2.set_xlim(0, 1)
    box2.set_ylim(0, 1)
    box2.set_facecolor("white")
    for sp in box2.spines.values():
        sp.set_edgecolor("#555555")
        sp.set_linewidth(1.3)
    box2.text(0.5, 0.93, "PCs that move pre -> post (BH over the first 10 PCs)", ha="center", va="center",
              fontsize=8.2, fontweight="bold")
    n = max(len(sig), 1)
    row_h = 0.74 / n
    d = post - pre
    for i, r in sig.iterrows():
        k = int(r.PC) - 1
        yc = 0.845 - (i + 0.5) * row_h
        high, low = PC_END_LABELS.get(int(r.PC), ("high end", "low end"))
        if sign[k] < 0:  # the map and this inset show the flipped PC, so its ends swap
            high, low = low, high
        box2.text(0.015, yc + 0.09, f"PC{int(r.PC)}" + (" (flipped)" if sign[k] < 0 else ""), fontsize=9,
                  fontweight="bold", va="center")
        box2.text(0.015, yc - 0.09, f"{100 * ev[k]:.1f}% var., p {r.p_fdr:.3f}", fontsize=7, va="center",
                  color="#333333")
        dd = box2.inset_axes([0.285, yc - 0.32 * row_h, 0.22, 0.64 * row_h])
        for q, cname in enumerate(CONDITIONS):
            m = lab == q
            mu = d[m, k].mean() / sd[k]
            bs = [d[m][rng.integers(0, m.sum(), m.sum()), k].mean() / sd[k] for _ in range(1000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            dd.errorbar(mu, q, xerr=[[mu - lo], [hi - mu]], fmt="o", ms=4.6, color=COLORS[cname], mec="white",
                        mew=0.4, ecolor=COLORS[cname], elinewidth=1.1, capsize=1.5)
        dd.axvline(0, color="#888888", lw=0.7)
        dd.set_ylim(2.6, -0.6)
        dd.set_xlim(-1.8, 1.4)
        dd.set_yticks([])
        dd.set_xticks([])
        for sp in ("left", "right", "top"):
            dd.spines[sp].set_visible(False)
        box2.text(0.535, yc + 0.09, "high: " + high, fontsize=8, va="center")
        box2.text(0.535, yc - 0.09, "low:  " + low, fontsize=8, va="center")
    box2.text(0.5, 0.035, "dots = mean change post - pre in SD units of the PC (bars: 95% CI); p = pooled BH-adjusted",
              fontsize=6.4, color="#666666", va="center", ha="center")
    save(fig, "prepost_semantic_space")


def pagerank():
    pr = pd.read_csv(OUT / "prepost_pagerank.csv")
    order = pr.groupby("topic").pagerank.mean().sort_values(ascending=False).index.tolist()
    ypos = {t: len(order) - 1 - i for i, t in enumerate(order)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True, sharex=True)
    for ax, c in zip(axes, CONDITIONS):
        for t in order:
            for time, off, filled in (("pre", 0.15, False), ("post", -0.15, True)):
                r = pr[(pr.condition == c) & (pr.time == time) & (pr.topic == t)].iloc[0]
                y = ypos[t] + off
                ax.errorbar(r.pagerank, y, xerr=[[r.pagerank - r.ci_low], [r.ci_high - r.pagerank]], fmt="o", ms=7,
                            mfc=DARK[c] if filled else "white", mec=DARK[c], mew=1.6, ecolor=COLORS[c],
                            elinewidth=1.3, capsize=2, zorder=3)
        tops = {tm: pr[(pr.condition == c) & (pr.time == tm)].sort_values("pagerank").iloc[-1] for tm in ("pre", "post")}
        n = {tm: int(pr[(pr.condition == c) & (pr.time == tm)].n_transitions.iloc[0]) for tm in ("pre", "post")}
        ax.axvline(1 / len(order), color="#888888", ls="--", lw=1)
        ax.set_title(f"{c}\ntop pre: {tops['pre'].topic}  |  top post: {tops['post'].topic}\n"
                     f"({n['pre']} pre / {n['post']} post transitions)", fontsize=10.5, color=DARK[c], fontweight="bold")
        ax.set_xlim(left=0)
        ax.set_xlabel("PageRank")
        ax.grid(axis="x", color="#e6e6e6")
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[t] for t in order])
    axes[0].set_yticklabels(order, fontsize=11)
    fig.suptitle("PageRank on the topic flow graph, pre (open) vs post (filled); bars = participant-bootstrap 95% CI; "
                 "dashed line = uniform (1/11).  Very few transitions per cell: descriptive only",
                 fontsize=11, y=1.03)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "prepost_pagerank")


def overview():
    ct = pd.read_csv(OUT / "prepost_content_tests.csv")
    pt = pd.read_csv(OUT / "prepost_pagerank_tests.csv")
    rows = []

    def get(df, analysis, test, comp):
        r = df[(df.analysis == analysis) & (df.test.str.startswith(test)) & (df.comparison == comp)]
        return None if r.empty else float(r.p_perm.iloc[0])

    cols = ["All", "VR Art", "VR Only", "Control"]
    base = "baseline (pre only)"
    time = "pre vs post (paired)"
    inter = "interaction (change differs by condition)"
    table = [
        ("Baseline: topic composition (shared topics)", [get(ct, base, "topic", "all three"), None, None, None]),
        ("Baseline: semantic space", [get(ct, base, "semantic", "all three"), None, None, None]),
        ("Baseline: PageRank (shared topics)", [get(pt, base, "GJSD", "all three"), None, None, None]),
        ("Pre->post: topic composition (shared topics)", [get(ct, time, "topic", c) for c in ["all conditions"] + CONDITIONS]),
        ("Pre->post: semantic space", [get(ct, time, "semantic", c) for c in ["all conditions"] + CONDITIONS]),
        ("Pre->post: path distributions, JSD (shared topics)", [None] + [get(pt, time, "JSD", c) for c in CONDITIONS]),
        ("Interaction: topic composition (shared topics)", [get(ct, inter, "topic", "all three"), None, None, None]),
        ("Interaction: semantic space", [get(ct, inter, "semantic", "all three"), None, None, None]),
        ("Interaction: PageRank change (shared topics)", [get(pt, inter, "PageRank change, omnibus", "all three"), None, None, None]),
    ]
    cmap = LinearSegmentedColormap.from_list("p", ["#c0392b", "#f5b7a5", "#ffffff"])
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(table) + 1.6))
    for i, (name, vals) in enumerate(table):
        for j, v in enumerate(vals):
            y = len(table) - 1 - i
            if v is None:
                continue
            ax.add_patch(plt.Rectangle((j, y), 1, 1, color=cmap(np.clip(v / 0.2, 0, 1)), ec="white", lw=2))
            ax.text(j + 0.5, y + 0.5, ("<.001" if v < 0.001 else f"{v:.3f}") + ("*" if v < 0.05 else ""), ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if v < 0.05 else "normal")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(table))
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_xticklabels(cols, fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(table)) + 0.5)
    ax.set_yticklabels([t[0] for t in table][::-1], fontsize=10.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Pre/post analysis: permutation p-values (uncorrected; per-topic and per-PC tests are in the CSVs)\n"
                 "* = p < .05.  Baseline and interaction: all three conditions together; pre->post: paired participants",
                 fontsize=10, y=1.12)
    save(fig, "prepost_tests_overview")


def between():
    r = pd.read_csv(OUT / "prepost_between_condition.csv")
    uni = r[r.type == "univariate"]
    measures = list(dict.fromkeys(uni.measure))
    cons = ["VR Art vs Control", "VR Only vs Control", "VR (Art + Only) vs Control", "VR Art vs VR Only"]
    colors = {"VR Art vs Control": DARK["VR Art"], "VR Only vs Control": DARK["VR Only"],
              "VR (Art + Only) vs Control": "#444444", "VR Art vs VR Only": "#999999"}
    offs = {c: o for c, o in zip(cons, (0.27, 0.09, -0.09, -0.27))}
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    for i, mname in enumerate(measures):
        y0 = len(measures) - 1 - i
        for c in cons:
            d = uni[(uni.measure == mname) & (uni.contrast == c)].iloc[0]
            ax.errorbar(d.std_difference, y0 + offs[c], xerr=[[d.std_difference - d.std_ci_low], [d.std_ci_high - d.std_difference]],
                        fmt="D" if "Art + Only" in c else "o", ms=7, color=colors[c], mec="white", mew=0.6,
                        elinewidth=1.5, capsize=2, zorder=3)
            padj = d.p_fdr_vr_vs_control if "Art + Only" in c else d.p_fdr_pairs
            ax.text(1.75, y0 + offs[c], f"p = {d.p_perm:.3f}  (adj. {padj:.3f})", va="center", fontsize=8.6,
                    color=colors[c])
    ax.axvline(0, color="#777777", lw=1)
    ax.set_xlim(-1.6, 3.05)
    ax.set_yticks(range(len(measures)))
    ax.set_yticklabels(measures[::-1], fontsize=10.5)
    ax.set_xlabel("Difference in mean change, first minus second condition (SD of participants' change); 95% bootstrap CI")
    ax.grid(axis="x", color="#e6e6e6")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for c in cons:
        ax.scatter([], [], color=colors[c], marker="D" if "Art + Only" in c else "o", label=c)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False, fontsize=9.5)
    mv = r[r.type == "multivariate"]
    txt = "   ".join(f"{m.split(' (')[0]}: " + ", ".join(
        f"{x.p_perm:.2f}" for x in mv[mv.measure == m].itertuples()) for m in list(dict.fromkeys(mv.measure)))
    fig.suptitle("Does the pre->post change differ between conditions?  (permutation p, condition labels shuffled; "
                 "adj. = BH-FDR over the 3 pairs; BH-FDR over measures for the VR vs Control contrast)\n"
                 "multivariate, p in the order VR Art vs VR Only | VR Art vs Control | VR Only vs Control | VR vs Control:  " + txt,
                 fontsize=9.6, y=0.995)
    fig.subplots_adjust(top=0.86)
    save(fig, "prepost_between_condition")


def path_jsd():
    """JSD between path distributions of the flow graphs: within condition (pre vs post), between conditions, interaction."""
    r = pd.read_csv(OUT / "prepost_path_jsd_tests.csv")
    r = r[(r.path_length == 3) & (r.smooth == 0.1)]
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.0), gridspec_kw={"width_ratios": [1, 1.25, 1, 1]})

    def draw(ax, sub, labels, colors, title, xlabel, ann_fdr=True):
        for k, (_, d) in enumerate(sub.iterrows()):
            y = len(sub) - 1 - k
            ax.plot([0, d.null_95], [y, y], lw=11, color="#dcdcdc", solid_capstyle="butt", zorder=1)
            ax.plot(d.null_mean, y, "|", color="#777777", ms=15, mew=2, zorder=2)
            ax.scatter(d.statistic, y, s=90, color=colors[k], edgecolor="#333333", lw=1.0, zorder=4)
            fdr = "" if pd.isna(d.p_fdr) else f", FDR {d.p_fdr:.2f}"
            ax.text(1.02, y, f"p = {d.p_perm:.2f}{fdr}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels[::-1], fontsize=10)
        ax.set_xlim(left=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.grid(axis="x", color="#e6e6e6")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    w = r[r.analysis.str.startswith("within")].set_index("comparison").loc[ORDER_C].reset_index()
    draw(axes[0], w, ORDER_C, [DARK[c] for c in ORDER_C], "Within condition:\npre vs post graph", "JSD (bits)")
    b = r[r.analysis.str.startswith("between") & (r.comparison != "all three")]
    rows_ = []
    labels_ = []
    cols_ = []
    for tm in ("pre", "post"):
        for pair in ("VR Art vs Control", "VR Only vs Control", "VR Art vs VR Only"):
            rows_.append(b[(b.analysis == f"between conditions ({tm})") & (b.comparison == pair)].iloc[0])
            labels_.append(f"{tm}: {pair}")
            cols_.append("#7f8fa6" if tm == "pre" else "#2f3542")
    draw(axes[1], pd.DataFrame(rows_), labels_, cols_, "Between conditions at one timepoint:\ngraph vs graph", "JSD (bits)")
    inter = r[r.analysis.str.startswith("interaction")]
    for ax, key, title, xl in ((axes[2], "amount", "Change differs? amount\n|JSD(A) - JSD(B)|", "difference in JSD (bits)"),
                               (axes[3], "direction", "Change differs? direction\nTV of change vectors", "TV distance")):
        sub = inter[inter.test.str.contains(key)].set_index("comparison").loc[["VR Art vs Control", "VR Only vs Control", "VR Art vs VR Only"]].reset_index()
        draw(ax, sub, list(sub.comparison), ["#555555"] * 3, title, xl)
    fig.suptitle("Jensen-Shannon divergence between the 3-step path distributions of the topic flow graphs "
                 "(dot = observed; grey bar = up to the 95th percentile under the null, tick = null mean; nothing exceeds chance)\n"
                 "within: pre/post labels swapped within participant; between and interaction: condition labels permuted across "
                 "participants; FDR = BH-FDR within family", fontsize=10.5, y=1.03)
    fig.tight_layout()
    save(fig, "prepost_path_jsd")


def topic_distance_detail():
    """Detail figure: all topic pairs, chance distribution and the A/B distances (topics extracted separately per timepoint)."""
    D = pd.read_csv(OUT / "topic_distance_matrix.csv", index_col=0)
    per = pd.read_csv(OUT / "topic_distance_per_topic.csv")
    tests = pd.read_csv(OUT / "topic_distance_tests.csv")
    z = np.load(OUT / "topic_distance_null.npz")
    fig = plt.figure(figsize=(32, 8.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 0.85, 1.0, 1.0], wspace=1.05)
    ax = fig.add_subplot(gs[0])
    sizes_pre = per[per.timepoint == "pre"].set_index("label").n_descriptors
    sizes_post = per[per.timepoint == "post"].set_index("label").n_descriptors
    im = ax.imshow(D.values, cmap="viridis_r", vmin=0, vmax=D.values.max(), aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            best_row = j == int(np.argmin(D.values[i]))
            best_col = i == int(np.argmin(D.values[:, j]))
            ax.text(j, i, f"{D.values[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if D.values[i, j] < 0.55 * D.values.max() else "#222222",
                    fontweight="bold" if (best_row or best_col) else "normal")
            if best_row:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#c0392b", lw=2.2))
            if best_col:
                ax.add_patch(plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, fill=False, ec="#f39c12", lw=1.6, ls="--"))
    ax.set_xticks(range(D.shape[1]))
    ax.set_xticklabels([f"{c} ({sizes_post[c]})" for c in D.columns], rotation=45, ha="left", fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(D.shape[0]))
    ax.set_yticklabels([f"{r} ({sizes_pre[r]})" for r in D.index], fontsize=9.5)
    ax.set_xlabel("post topics (number of descriptors)", fontsize=10.5)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("pre topics (number of descriptors)", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="cosine distance between topic centres")
    ax.text(0.5, -0.045, "Distance between every pre and post topic.\nRed box = closest post topic to a pre topic; "
            "dashed = closest pre topic to a post topic", transform=ax.transAxes, ha="center", va="top", fontsize=9.5)
    S = tests[tests.analysis.str.startswith("set distance S")].iloc[0]
    M = tests[tests.analysis.str.startswith("set distance M")].iloc[0]
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(z["S"], bins=40, color="#cfcfcf", edgecolor="white", label="chance: S with pre/post swapped within participants")
    ax2.axvline(S.statistic, color="#c0392b", lw=2.4, label=f"observed S = {S.statistic:.3f}")
    ax2.set_yticks([])
    ax2.set_xlabel("set distance S (size-weighted nearest-counterpart distance)")
    ax2.set_title(f"Are the pre topics further from the post topics\nthan chance?  p = {S.p_perm:.3f} (M = {M.statistic:.3f}, p = {M.p_perm:.3f})",
                  fontsize=10.5)
    ax2.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax3 = fig.add_subplot(gs[2])
    cs = tests[tests.analysis.str.startswith("set distance S within")].set_index("comparison")
    ca = tests[tests.analysis.str.startswith("post descriptors")].set_index("comparison")
    cb = tests[tests.analysis.str.startswith("pre descriptors")].set_index("comparison")
    for k, c in enumerate(CONDITIONS):
        for off, tb, nm, mk in ((0.25, cs, "S", "D"), (0.0, ca, "A", "o"), (-0.25, cb, "B", "s")):
            r = tb.loc[c]
            y = (2 - k) + off
            ax3.plot([0, r.null_95], [y, y], lw=8, color="#dcdcdc", solid_capstyle="butt", zorder=1)
            ax3.plot(r.null_mean, y, "|", color="#777777", ms=12, mew=2, zorder=2)
            ax3.scatter(r.statistic, y, s=75, color=DARK[c], edgecolor="#333333", lw=1, zorder=3, marker=mk)
            ax3.text(1.03, y, f"{nm}: p = {r.p_perm:.3f}, FDR {r.p_fdr:.3f}", transform=ax3.get_yaxis_transform(), va="center", fontsize=8.5)
    ax3.set_yticks([2, 1, 0])
    ax3.set_yticklabels(CONDITIONS, fontsize=11)
    ax3.set_xlim(left=0)
    ax3.set_xlabel("distance to the nearest topic of the other timepoint")
    ax3.set_title("Within each condition\nS (diamond) = topic-set distance with the condition's own topic centres;\nA (circle) = post descriptors to nearest pre topic; B (square) = pre descriptors to nearest post topic\n(grey bar = up to the 95th percentile under chance, tick = chance mean)",
                  fontsize=9)
    ax3.grid(axis="x", color="#e6e6e6")
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)
    ax4 = fig.add_subplot(gs[3])
    rows_, ylabs = [], []
    for key, nm in (("S: set distance", "S"), ("A: post descriptors", "A"), ("B: pre descriptors", "B")):
        sub = tests[tests.analysis.str.startswith("between conditions, " + key)].set_index("comparison")
        for pair in ("VR Art vs Control", "VR Only vs Control", "VR Art vs VR Only"):
            rows_.append(sub.loc[pair])
            ylabs.append(f"{nm}: {pair}")
    for k, r in enumerate(rows_):
        y = len(rows_) - 1 - k
        ax4.plot([-r.null_95, r.null_95], [y, y], lw=8, color="#dcdcdc", solid_capstyle="butt", zorder=1)
        ax4.scatter(r.statistic, y, s=70, color="#444444", edgecolor="white", lw=0.8, zorder=3)
        ax4.text(1.03, y, f"p = {r.p_perm:.2f}, FDR {r.p_fdr:.2f}", transform=ax4.get_yaxis_transform(), va="center", fontsize=8.5)
    ax4.axvline(0, color="#777777", lw=1)
    ax4.set_yticks(range(len(rows_)))
    ax4.set_yticklabels(ylabs[::-1], fontsize=9.5)
    ax4.set_xlabel("difference between the two conditions (first minus second)")
    ax4.set_title("Between conditions\ndot = observed difference; grey bar = 95% of the differences\nwhen condition labels are shuffled across participants", fontsize=9)
    ax4.grid(axis="x", color="#e6e6e6")
    ax4.set_axisbelow(True)
    for sp in ("top", "right"):
        ax4.spines[sp].set_visible(False)
    fig.suptitle("How far are the topics extracted from the pre interview from those extracted from the post interview? "
                 "(topics fitted separately per timepoint, compared in the embedding space)", fontsize=12, y=1.13)
    save(fig, "topic_distance_detail")


def topic_distance():
    """Main figure: topic-set distance S between the pre and post topics, within and between conditions."""
    tests = pd.read_csv(OUT / "topic_distance_tests.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.75})
    ov = tests[tests.analysis.str.startswith("set distance S (")].iloc[0]
    cs = tests[tests.analysis.str.startswith("set distance S within")].set_index("comparison")
    rows = [("All participants", ov, "#444444")] + [(c, cs.loc[c], DARK[c]) for c in CONDITIONS]
    for k, (name, r, col) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot([0, r.null_95], [y, y], lw=13, color="#dcdcdc", solid_capstyle="butt", zorder=1)
        ax.plot(r.null_mean, y, "|", color="#777777", ms=17, mew=2.2, zorder=2)
        ax.scatter(r.statistic, y, s=130, marker="D", color=col, edgecolor="white", lw=1.2, zorder=3)
        fdr = "" if pd.isna(r.p_fdr) else f", FDR {r.p_fdr:.3f}"
        ax.text(1.03, y, f"p = {r.p_perm:.3f}{fdr}", transform=ax.get_yaxis_transform(), va="center", fontsize=10,
                fontweight="bold" if (r.p_perm < 0.05 if pd.isna(r.p_fdr) else r.p_fdr < 0.05) else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=11.5)
    ax.set_xlim(left=0)
    ax.set_xlabel("topic-set distance S (cosine distance to the nearest topic of the other timepoint, size-weighted)")
    ax.set_title("Within: how far are the post topics from the pre topics?\n"
                 "diamond = observed; grey bar = up to the 95th percentile expected by chance, tick = chance mean\n"
                 "(chance: pre/post labels swapped within participants, topics refitted 1000 times)", fontsize=9.6)
    ax.grid(axis="x", color="#e6e6e6")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    pairs = ("VR Art vs Control", "VR Only vs Control", "VR Art vs VR Only")
    bt = tests[tests.analysis.str.startswith("between conditions, S:")].set_index("comparison")
    for k, pair in enumerate(pairs):
        r = bt.loc[pair]
        y = len(pairs) - 1 - k
        ax2.plot([-r.null_95, r.null_95], [y, y], lw=13, color="#dcdcdc", solid_capstyle="butt", zorder=1)
        ax2.scatter(r.statistic, y, s=110, color="#444444", edgecolor="white", lw=1, zorder=3)
        ax2.text(1.03, y, f"p = {r.p_perm:.3f}, FDR {r.p_fdr:.3f}", transform=ax2.get_yaxis_transform(), va="center", fontsize=10)
    ax2.axvline(0, color="#777777", lw=1)
    ax2.set_yticks(range(len(pairs)))
    ax2.set_yticklabels(pairs[::-1], fontsize=11.5)
    ax2.set_xlabel("difference in S between the two conditions (first minus second)")
    ax2.set_title("Between: do the conditions differ in how far their topics move?\n"
                  "dot = observed difference; grey bar = 95% of differences when condition labels\nare shuffled across participants (topics fixed)",
                  fontsize=9.6)
    ax2.grid(axis="x", color="#e6e6e6")
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    fig.suptitle("Distance between the topics extracted from the pre interview and those extracted from the post interview "
                 "(topics fitted separately per timepoint)", fontsize=11.5, y=1.08)
    save(fig, "topic_distance")


ORDER_C = ["VR Art", "VR Only", "Control"]


if __name__ == "__main__":
    topic_shares()
    semantic_space()
    pagerank()
    overview()
    between()
    path_jsd()
    topic_distance()
    topic_distance_detail()
