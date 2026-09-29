#!/usr/bin/env python3
"""Bubble version of the condition topic plots: one dot per condition x topic,
dot area = proportion of that condition's sentences/responses in the topic."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from all_questions.response_topic_labels import short_response_topic_label  # noqa: E402
from all_questions.topic_labels import short_topic_label  # noqa: E402
from q3.plot_topic_differences import CONDITION_COLORS, CONDITION_ORDER  # noqa: E402

DEFAULT_OUTPUT = BASE_DIR / "all_questions" / "outputs"
RUNS = [
    ("combined_questions_sentence_topic_plot_data.csv",
     "combined_questions_sentence_topic_bubbles_by_condition", "% of condition's sentences",
     short_topic_label),
    ("combined_responses_raw_plot_data.csv",
     "combined_responses_topic_bubbles_raw", "% of condition's responses",
     short_response_topic_label),
    ("combined_responses_participant_weighted_plot_data.csv",
     "combined_responses_topic_bubbles_participant_weighted",
     "mean % of participant's responses", short_response_topic_label),
]
MAX_AREA = 1500  # marker area (pt^2) for the largest dot, shared across plots
LEGEND_PCTS = [5, 15, 30]


def draw(data: pd.DataFrame, out: Path, stem: str, unit: str, vmax: float, label_fn) -> None:
    order = (data[["Topic", "topic_label", "difference_rank"]].drop_duplicates()
             .sort_values("difference_rank"))
    ypos = {t: i for i, t in enumerate(order["Topic"])}
    xpos = {c: i for i, c in enumerate(CONDITION_ORDER)}
    fig, ax = plt.subplots(figsize=(8, max(5.5, 0.62 * len(order) + 2)))
    for r in data.itertuples():
        if r.condition_label not in xpos or r.mean_percent <= 0:
            continue
        ax.scatter(xpos[r.condition_label], ypos[r.Topic],
                   s=MAX_AREA * r.mean_percent / vmax,
                   color=CONDITION_COLORS[r.condition_label],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(xpos[r.condition_label], ypos[r.Topic], f"{r.mean_percent:.0f}",
                ha="center", va="center", fontsize=8, color="#333333", zorder=4)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([label_fn(t) for t in order["Topic"]], fontsize=12)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(CONDITION_ORDER, fontsize=13)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xlim(-0.6, len(xpos) - 0.4)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.grid(color="#e4e4e4", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=(MAX_AREA * p / vmax) ** 0.5,
                      markerfacecolor="#bbbbbb", markeredgecolor="white", label=f"{p}%")
               for p in LEGEND_PCTS]
    ax.legend(handles=handles, title=unit, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=len(handles), frameon=False,
              labelspacing=1.6, handletextpad=1.0, columnspacing=2.5, borderpad=1.5)
    base = out / stem
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(base.with_suffix(f".{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"Wrote {base}.[pdf|svg|png]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    out = p.parse_args().output_dir.expanduser().resolve()
    for csv, stem, unit, label_fn in RUNS:
        draw(pd.read_csv(out / csv), out, stem, unit, vmax=35.0, label_fn=label_fn)


if __name__ == "__main__":
    main()
