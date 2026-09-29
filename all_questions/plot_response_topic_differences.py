#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from all_questions.plot_topic_differences import draw_plot
from all_questions.response_topic_labels import response_topic_label


DEFAULT_OUTPUT = BASE_DIR / "all_questions" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot combined-response BERTopic prevalence by condition."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-outlier", action="store_true")
    return parser.parse_args()


def prepare_participant_plot(
    prevalence: pd.DataFrame, include_outlier: bool
) -> pd.DataFrame:
    data = prevalence.copy()
    if not include_outlier:
        data = data.loc[data["Topic"] != -1].copy()
    data["topic_label"] = data["Topic"].map(response_topic_label)
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


def prepare_raw_plot(prevalence: pd.DataFrame, include_outlier: bool) -> pd.DataFrame:
    data = prevalence.copy()
    if not include_outlier:
        data = data.loc[data["Topic"] != -1].copy()
    data["topic_label"] = data["Topic"].map(response_topic_label)
    data["mean_percent"] = 100 * data["proportion_within_condition"]
    data["error_lower"] = data["mean_percent"]
    data["error_upper"] = data["mean_percent"]
    ranges = data.groupby("Topic")["mean_percent"].agg(
        lambda values: values.max() - values.min()
    ).sort_values(ascending=False)
    data["difference_rank"] = data["Topic"].map(
        {topic: rank for rank, topic in enumerate(ranges.index)}
    )
    return data.sort_values(["difference_rank", "condition_number"])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    weighted = pd.read_csv(
        output_dir
        / "combined_responses_topic_prevalence_by_condition_participant_weighted.csv"
    )
    weighted_plot = prepare_participant_plot(weighted, args.include_outlier)
    weighted_plot.to_csv(
        output_dir / "combined_responses_participant_weighted_plot_data.csv",
        index=False,
    )
    draw_plot(
        weighted_plot,
        output_dir,
        "combined_responses_topic_differences_participant_weighted",
        x_label="Mean participant response-topic prevalence (%)",
    )

    raw = pd.read_csv(
        output_dir / "combined_responses_topic_prevalence_by_condition.csv"
    )
    raw_plot = prepare_raw_plot(raw, args.include_outlier)
    raw_plot.to_csv(
        output_dir / "combined_responses_raw_plot_data.csv", index=False
    )
    draw_plot(
        raw_plot,
        output_dir,
        "combined_responses_topic_differences_raw",
        x_label="Response-level topic prevalence (%)",
    )
    print(f"Response-level plots written to: {output_dir}")


if __name__ == "__main__":
    main()
