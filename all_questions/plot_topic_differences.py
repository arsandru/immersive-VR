#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from q3.plot_topic_differences import (  # noqa: E402
    CONDITION_COLORS,
    CONDITION_OFFSETS,
    CONDITION_ORDER,
)
from all_questions.topic_labels import topic_label  # noqa: E402


DEFAULT_OUTPUT = BASE_DIR / "all_questions" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot participant-weighted topic prevalence for all questions."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-outlier", action="store_true")
    return parser.parse_args()


def prepare_plot_data(
    prevalence: pd.DataFrame,
    topic_info: pd.DataFrame,
    include_outlier: bool,
) -> pd.DataFrame:
    data = prevalence.copy()
    if not include_outlier:
        data = data.loc[data["Topic"] != -1].copy()
    labels = {
        int(row.Topic): topic_label(int(row.Topic))
        for row in topic_info.itertuples(index=False)
    }
    data["topic_label"] = data["Topic"].map(labels)
    data["mean_percent"] = 100 * data["mean_proportion"]
    data["error_half_width"] = 100 * data["sd_proportion"].fillna(0)
    data["error_lower"] = (data["mean_percent"] - data["error_half_width"]).clip(lower=0)
    data["error_upper"] = (data["mean_percent"] + data["error_half_width"]).clip(upper=100)
    ranges = data.groupby("Topic")["mean_percent"].agg(
        lambda values: values.max() - values.min()
    ).sort_values(ascending=False)
    data["difference_rank"] = data["Topic"].map(
        {topic: rank for rank, topic in enumerate(ranges.index)}
    )
    return data.sort_values(["difference_rank", "condition_number"])


def prepare_sentence_plot_data(
    prevalence: pd.DataFrame,
    topic_info: pd.DataFrame,
    include_outlier: bool,
) -> pd.DataFrame:
    data = prevalence.copy()
    if not include_outlier:
        data = data.loc[data["Topic"] != -1].copy()
    labels = {
        int(row.Topic): topic_label(int(row.Topic))
        for row in topic_info.itertuples(index=False)
    }
    data["topic_label"] = data["Topic"].map(labels)
    data["mean_percent"] = 100 * data["proportion_within_group"]
    data["error_lower"] = data["mean_percent"]
    data["error_upper"] = data["mean_percent"]
    ranges = data.groupby("Topic")["mean_percent"].agg(
        lambda values: values.max() - values.min()
    ).sort_values(ascending=False)
    data["difference_rank"] = data["Topic"].map(
        {topic: rank for rank, topic in enumerate(ranges.index)}
    )
    return data.sort_values(["difference_rank", "condition_number"])


def draw_plot(
    data: pd.DataFrame,
    output_dir: Path,
    stem: str,
    x_label: str = "Mean participant topic prevalence (%)",
) -> None:
    ordered = data[["Topic", "topic_label", "difference_rank"]].drop_duplicates().sort_values(
        "difference_rank"
    )
    positions = {
        int(topic): position
        for position, topic in enumerate(ordered["Topic"].tolist()[::-1])
    }
    fig, ax = plt.subplots(figsize=(11, max(6.5, 0.62 * len(ordered) + 1.8)))
    for condition in CONDITION_ORDER:
        condition_data = data.loc[data["condition_label"] == condition].copy()
        condition_data["y"] = condition_data["Topic"].map(positions) + CONDITION_OFFSETS[condition]
        lower = condition_data["mean_percent"] - condition_data["error_lower"]
        upper = condition_data["error_upper"] - condition_data["mean_percent"]
        ax.errorbar(
            condition_data["mean_percent"],
            condition_data["y"],
            xerr=np.vstack([lower, upper]),
            fmt="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=CONDITION_COLORS[condition],
            ecolor=CONDITION_COLORS[condition],
            elinewidth=1.8,
            capsize=3,
            label=condition,
            zorder=3,
        )
    labels = ordered.set_index("Topic")["topic_label"].to_dict()
    position_topics = {position: topic for topic, position in positions.items()}
    ticks = sorted(position_topics)
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[position_topics[tick]] for tick in ticks])
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#8a8a8a")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    ax.set_xlim(0, max(10, min(100, data["error_upper"].max() * 1.08)))
    fig.subplots_adjust(left=0.34, right=0.97, top=0.91, bottom=0.11)
    base = output_dir / stem
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    topic_info = pd.read_csv(output_dir / "all_questions_topics.csv")

    combined = pd.read_csv(
        output_dir / "combined_questions_sentence_topic_prevalence_by_condition.csv"
    )
    combined_plot = prepare_sentence_plot_data(
        combined, topic_info, args.include_outlier
    )
    combined_plot.to_csv(
        output_dir / "combined_questions_sentence_topic_plot_data.csv", index=False
    )
    draw_plot(
        combined_plot,
        output_dir,
        "combined_questions_sentence_topic_differences_by_condition",
        x_label="Sentence-level topic prevalence (%)",
    )

    overall = pd.read_csv(
        output_dir / "all_questions_topic_prevalence_by_condition_participant_weighted.csv"
    )
    overall_plot = prepare_plot_data(overall, topic_info, args.include_outlier)
    overall_plot.to_csv(output_dir / "all_questions_topic_plot_data.csv", index=False)
    draw_plot(overall_plot, output_dir, "all_questions_topic_differences_by_condition")

    by_question = pd.read_csv(
        output_dir
        / "all_questions_topic_prevalence_by_question_condition_participant_weighted.csv"
    )
    question_plot_frames = []
    for question_id, frame in by_question.groupby("question_id", sort=True):
        frame = frame.loc[
            frame.groupby("Topic")["mean_proportion"].transform("max") > 0
        ].copy()
        plot_data = prepare_plot_data(frame, topic_info, args.include_outlier)
        question_plot_frames.append(plot_data)
        draw_plot(
            plot_data,
            output_dir,
            f"all_questions_topic_differences_by_condition_{question_id.lower()}",
        )
    pd.concat(question_plot_frames, ignore_index=True).to_csv(
        output_dir / "all_questions_topic_plot_data_by_question.csv", index=False
    )
    print(f"Plots written to: {output_dir}")


if __name__ == "__main__":
    main()
