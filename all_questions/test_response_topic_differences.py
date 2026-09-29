#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from all_questions.response_topic_labels import response_topic_label
from all_questions.test_topic_differences import bh_adjust, holm_adjust, test_topics


DEFAULT_OUTPUT = BASE_DIR / "all_questions" / "outputs"
CONDITIONS = ["VR Art", "VR Only", "Control"]
PAIRWISE_CONTRASTS = [
    ("VR Art", "Control"),
    ("VR Art", "VR Only"),
    ("VR Only", "Control"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test participant-level response-topic proportions by condition."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def pairwise_followups(data: pd.DataFrame, omnibus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    significant_topics = omnibus.loc[omnibus["bh_fdr_p_value"] < 0.05, "Topic"]
    for topic in significant_topics:
        topic_data = data.loc[data["Topic"] == topic]
        topic_rows = []
        for first, second in PAIRWISE_CONTRASTS:
            first_values = topic_data.loc[
                topic_data["condition_label"] == first,
                "participant_topic_proportion",
            ].to_numpy()
            second_values = topic_data.loc[
                topic_data["condition_label"] == second,
                "participant_topic_proportion",
            ].to_numpy()
            statistic, raw_p = mannwhitneyu(
                first_values,
                second_values,
                alternative="two-sided",
                method="asymptotic",
            )
            rank_biserial = 2 * statistic / (len(first_values) * len(second_values)) - 1
            topic_rows.append(
                {
                    "Topic": int(topic),
                    "topic_label": response_topic_label(int(topic)),
                    "comparison": f"{first} vs {second}",
                    "first_condition": first,
                    "second_condition": second,
                    "n_first": len(first_values),
                    "n_second": len(second_values),
                    "mean_first": first_values.mean(),
                    "mean_second": second_values.mean(),
                    "mann_whitney_u": statistic,
                    "rank_biserial": rank_biserial,
                    "raw_p_value": raw_p,
                }
            )
        frame = pd.DataFrame(topic_rows)
        frame["holm_p_value"] = holm_adjust(frame["raw_p_value"].to_numpy())
        frame["bh_fdr_p_value"] = bh_adjust(frame["raw_p_value"].to_numpy())
        rows.append(frame)
    if not rows:
        return pd.DataFrame(
            columns=[
                "Topic", "topic_label", "comparison", "first_condition",
                "second_condition", "n_first", "n_second", "mean_first",
                "mean_second", "mann_whitney_u", "rank_biserial",
                "raw_p_value", "holm_p_value", "bh_fdr_p_value",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    data = pd.read_csv(
        output_dir / "combined_responses_participant_topic_proportions.csv"
    )
    # Reuse the generic topic test, which checks this count column for absent topics.
    data["n_sentences"] = data["n_responses"]
    labels = {
        int(topic): response_topic_label(int(topic))
        for topic in data["Topic"].unique()
    }
    tests = test_topics(data, labels, family_columns=[])
    tests.to_csv(
        output_dir / "combined_responses_topic_condition_tests.csv", index=False
    )
    pairwise = pairwise_followups(data, tests)
    pairwise.to_csv(
        output_dir / "combined_responses_topic_pairwise_tests.csv", index=False
    )
    print(f"FDR-significant response-topic tests: {(tests.bh_fdr_p_value < 0.05).sum()}")
    print(f"FDR-significant gated pairwise tests: {(pairwise.bh_fdr_p_value < 0.05).sum()}")


if __name__ == "__main__":
    main()
