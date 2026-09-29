#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from q3.plot_topic_differences import parse_representation  # noqa: E402
from all_questions.topic_labels import topic_label as human_topic_label  # noqa: E402


DEFAULT_OUTPUT = BASE_DIR / "all_questions" / "outputs"
CONDITION_ORDER = ["VR Art", "VR Only", "Control"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test participant-level topic-proportion differences by condition."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_min = 1.0
    for zero_rank in range(len(p_values) - 1, -1, -1):
        index = order[zero_rank]
        rank = zero_rank + 1
        running_min = min(running_min, p_values[index] * len(p_values) / rank)
        adjusted[index] = min(1.0, running_min)
    return adjusted


def topic_label(topic: int, representation: object) -> str:
    terms = parse_representation(representation)
    return " / ".join(terms[:3]) or f"Topic {topic}"


def test_topics(
    data: pd.DataFrame,
    topic_labels: dict[int, str],
    family_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = data.loc[data["Topic"] != -1].groupby(family_columns + ["Topic"], dropna=False)
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        topic = int(keys[-1])
        family_values = keys[:-1]
        if frame["n_sentences"].sum() == 0:
            continue
        groups = [
            frame.loc[
                frame["condition_label"] == condition,
                "participant_topic_proportion",
            ].to_numpy()
            for condition in CONDITION_ORDER
        ]
        if any(len(group) == 0 for group in groups):
            statistic, raw_p = np.nan, np.nan
        elif np.ptp(np.concatenate(groups)) == 0:
            statistic, raw_p = 0.0, 1.0
        else:
            statistic, raw_p = kruskal(*groups)

        row = {
            column: value for column, value in zip(family_columns, family_values)
        }
        row.update(
            {
                "Topic": topic,
                "topic_label": topic_labels.get(topic, f"Topic {topic}"),
                "kruskal_h": statistic,
                "df": len(CONDITION_ORDER) - 1,
                "raw_p_value": raw_p,
            }
        )
        for condition, values in zip(CONDITION_ORDER, groups):
            slug = condition.lower().replace("+", "_plus_").replace(" ", "_")
            row[f"n_{slug}"] = len(values)
            row[f"mean_{slug}"] = float(np.mean(values)) if len(values) else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    correction_groups = result.groupby(family_columns, dropna=False) if family_columns else [(None, result)]
    corrected_frames = []
    for _, family in correction_groups:
        family = family.copy()
        valid = family["raw_p_value"].notna()
        family["holm_p_value"] = np.nan
        family["bh_fdr_p_value"] = np.nan
        if valid.any():
            values = family.loc[valid, "raw_p_value"].to_numpy()
            family.loc[valid, "holm_p_value"] = holm_adjust(values)
            family.loc[valid, "bh_fdr_p_value"] = bh_adjust(values)
        corrected_frames.append(family)
    return pd.concat(corrected_frames, ignore_index=True).sort_values(
        family_columns + ["raw_p_value"]
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    topic_info = pd.read_csv(output_dir / "all_questions_topics.csv")
    labels = {
        int(row.Topic): human_topic_label(int(row.Topic))
        for row in topic_info.itertuples(index=False)
    }

    overall = pd.read_csv(output_dir / "all_questions_participant_topic_proportions.csv")
    overall_tests = test_topics(overall, labels, family_columns=[])
    overall_tests.to_csv(
        output_dir / "all_questions_topic_condition_tests.csv", index=False
    )

    by_question = pd.read_csv(
        output_dir / "all_questions_participant_question_topic_proportions.csv"
    )
    question_tests = test_topics(
        by_question,
        labels,
        family_columns=["question_id", "question_text"],
    )
    question_tests.to_csv(
        output_dir / "all_questions_topic_condition_tests_by_question.csv", index=False
    )

    n_overall = int((overall_tests["bh_fdr_p_value"] < 0.05).sum())
    n_question = int((question_tests["bh_fdr_p_value"] < 0.05).sum())
    print(f"FDR-significant overall topic tests: {n_overall}")
    print(f"FDR-significant question-specific topic tests: {n_question}")


if __name__ == "__main__":
    main()
