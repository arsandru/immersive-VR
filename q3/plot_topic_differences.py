#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "q3" / "outputs"

CONDITION_ORDER = ["VR Art", "VR Only", "Control"]
CONDITION_COLORS = {
    "VR Art": "#8de5a1",
    "VR Only": "#ffb482",
    "Control": "#a1c9f4",
}
CONDITION_OFFSETS = {
    "VR Art": -0.22,
    "VR Only": 0.0,
    "Control": 0.22,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot participant-weighted BERTopic prevalence by condition."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-outlier",
        action="store_true",
        help="Include BERTopic topic -1 (unassigned/outlier sentences).",
    )
    return parser.parse_args()


def parse_representation(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = [part.strip() for part in value.split(",")]
    return [str(item).strip() for item in parsed if str(item).strip()]


def representative_label(topic: int, representation: object, limit: int = 3) -> str:
    if topic == -1:
        return "Unassigned / outlier"

    selected: list[str] = []
    for candidate in parse_representation(representation):
        candidate_tokens = set(candidate.casefold().split())
        if not candidate_tokens:
            continue

        replaced = False
        for index, existing in enumerate(selected):
            existing_tokens = set(existing.casefold().split())
            if candidate_tokens == existing_tokens or candidate_tokens < existing_tokens:
                replaced = True
                break
            if existing_tokens < candidate_tokens:
                selected[index] = candidate
                replaced = True
                break

        if not replaced:
            selected.append(candidate)
        if len(selected) >= limit:
            break

    words = " · ".join(selected[:limit]) or f"Topic {topic}"
    return f"{topic}: {words}"


def prepare_plot_data(
    prevalence: pd.DataFrame,
    topic_info: pd.DataFrame,
    include_outlier: bool,
) -> pd.DataFrame:
    data = prevalence.copy()
    if not include_outlier:
        data = data.loc[data["Topic"] != -1].copy()

    labels = {
        int(row.Topic): representative_label(int(row.Topic), row.Representation)
        for row in topic_info.itertuples(index=False)
    }
    data["topic_label"] = data["Topic"].map(labels)
    data["mean_percent"] = 100 * data["mean_proportion"]
    data["ci_half_width"] = 100 * 1.96 * data["se_proportion"].fillna(0)
    data["ci_lower"] = (data["mean_percent"] - data["ci_half_width"]).clip(lower=0)
    data["ci_upper"] = (data["mean_percent"] + data["ci_half_width"]).clip(upper=100)

    topic_ranges = (
        data.groupby("Topic")["mean_percent"]
        .agg(lambda values: values.max() - values.min())
        .sort_values(ascending=False)
    )
    rank = {topic: index for index, topic in enumerate(topic_ranges.index)}
    data["difference_rank"] = data["Topic"].map(rank)
    return data.sort_values(["difference_rank", "condition_number"])


def plot_topic_differences(data: pd.DataFrame, output_dir: Path) -> None:
    ordered_topics = (
        data[["Topic", "topic_label", "difference_rank"]]
        .drop_duplicates()
        .sort_values("difference_rank")
    )
    topic_positions = {
        int(topic): position
        for position, topic in enumerate(ordered_topics["Topic"].tolist()[::-1])
    }

    figure_height = max(6.5, 0.62 * len(ordered_topics) + 1.8)
    fig, ax = plt.subplots(figsize=(11, figure_height))

    for condition in CONDITION_ORDER:
        condition_data = data.loc[data["condition_label"] == condition].copy()
        condition_data["y"] = (
            condition_data["Topic"].map(topic_positions) + CONDITION_OFFSETS[condition]
        )
        lower_error = condition_data["mean_percent"] - condition_data["ci_lower"]
        upper_error = condition_data["ci_upper"] - condition_data["mean_percent"]
        ax.errorbar(
            condition_data["mean_percent"],
            condition_data["y"],
            xerr=np.vstack([lower_error, upper_error]),
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

    labels_by_topic = ordered_topics.set_index("Topic")["topic_label"].to_dict()
    y_ticks = sorted(topic_positions.values())
    topic_by_position = {position: topic for topic, position in topic_positions.items()}
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([labels_by_topic[topic_by_position[pos]] for pos in y_ticks])
    ax.set_xlabel("Mean participant topic prevalence (%)", fontsize=14)
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
        handletextpad=0.5,
        columnspacing=1.8,
    )

    max_ci = data["ci_upper"].max()
    ax.set_xlim(0, max(10, min(100, max_ci * 1.08)))
    fig.subplots_adjust(left=0.34, right=0.97, top=0.91, bottom=0.11)

    base_path = output_dir / "q3_topic_differences_by_condition"
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    prevalence_path = output_dir / "q3_topic_prevalence_by_condition_participant_weighted.csv"
    topics_path = output_dir / "q3_topics.csv"

    if not prevalence_path.exists() or not topics_path.exists():
        raise SystemExit(
            "Required q3 outputs are missing. Run q3/q3_topic_analysis.py first."
        )

    prevalence = pd.read_csv(prevalence_path)
    topic_info = pd.read_csv(topics_path)
    plot_data = prepare_plot_data(prevalence, topic_info, args.include_outlier)
    plot_data.to_csv(output_dir / "q3_topic_plot_data.csv", index=False)
    plot_topic_differences(plot_data, output_dir)
    print(f"Plot written to: {output_dir / 'q3_topic_differences_by_condition.pdf'}")


if __name__ == "__main__":
    main()
