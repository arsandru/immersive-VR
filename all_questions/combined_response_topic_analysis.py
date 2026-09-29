#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from all_questions.all_questions_topic_analysis import load_long_responses  # noqa: E402
from all_questions.response_topic_labels import response_topic_label  # noqa: E402
from q3.q3_topic_analysis import (  # noqa: E402
    MODEL_ID,
    PORTUGUESE_STOP_WORDS,
    content_tokens,
    package_version,
)


EMBED_INSTRUCTION = (
    "Instruct: Represent the main topic and semantic meaning of this Portuguese "
    "interview response.\nText: "
)
DEFAULT_CONDITIONS = BASE_DIR / "data" / "participant_conditions.csv"
DEFAULT_INTERVIEWS = BASE_DIR / "data" / "immerse_interviewPost.csv"
DEFAULT_OUTPUT = BASE_DIR / "all_questions" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit BERTopic to complete responses from all three interview questions."
        )
    )
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--interviews", type=Path, default=DEFAULT_INTERVIEWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--cluster-selection-method", choices=("leaf", "eom"), default="leaf"
    )
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def prepare_responses(responses: pd.DataFrame) -> pd.DataFrame:
    data = responses.copy()
    data["content_text"] = data["response"].map(
        lambda text: " ".join(content_tokens(text))
    )
    data["content_token_count"] = data["content_text"].str.split().str.len().fillna(0).astype(int)
    data["included_in_response_model"] = (
        data["has_response"] & data["has_condition"] & data["content_token_count"].gt(0)
    )
    data["response_model_exclusion_reason"] = np.select(
        [
            ~data["has_response"],
            ~data["has_condition"],
            data["content_token_count"].eq(0),
        ],
        [
            "blank response",
            "condition not matched",
            "no substantive tokens after stopword removal",
        ],
        default="",
    )
    return data


def raw_response_prevalence(
    documents: pd.DataFrame,
    topic_names: dict[int, str],
) -> pd.DataFrame:
    result = documents.groupby(
        ["condition_number", "condition_label", "Topic"], dropna=False
    ).size().rename("n_responses").reset_index()
    result["n_condition_responses"] = result.groupby("condition_number")[
        "n_responses"
    ].transform("sum")
    result["proportion_within_condition"] = (
        result["n_responses"] / result["n_condition_responses"]
    )
    result["topic_name"] = result["Topic"].map(topic_names)
    result["topic_label"] = result["Topic"].map(response_topic_label)
    return result.sort_values(["condition_number", "Topic"])


def participant_response_prevalence(
    documents: pd.DataFrame,
    topic_names: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = documents[
        ["participant_id", "condition_number", "condition_label"]
    ].drop_duplicates()
    topics = pd.DataFrame({"Topic": sorted(documents["Topic"].unique())})
    grid = participants.merge(topics, how="cross")
    counts = documents.groupby(
        ["participant_id", "condition_number", "condition_label", "Topic"]
    ).size().rename("n_responses").reset_index()
    totals = documents.groupby("participant_id").size().rename(
        "n_participant_responses"
    ).reset_index()
    participant_topics = grid.merge(
        counts,
        on=["participant_id", "condition_number", "condition_label", "Topic"],
        how="left",
    ).merge(totals, on="participant_id", how="left")
    participant_topics["n_responses"] = participant_topics["n_responses"].fillna(0).astype(int)
    participant_topics["participant_topic_proportion"] = (
        participant_topics["n_responses"]
        / participant_topics["n_participant_responses"]
    )
    participant_topics["topic_name"] = participant_topics["Topic"].map(topic_names)
    participant_topics["topic_label"] = participant_topics["Topic"].map(
        response_topic_label
    )
    summary = participant_topics.groupby(
        ["condition_number", "condition_label", "Topic", "topic_name", "topic_label"],
        dropna=False,
    )["participant_topic_proportion"].agg(
        n_participants="size",
        mean_proportion="mean",
        sd_proportion="std",
    ).reset_index()
    summary["se_proportion"] = summary["sd_proportion"] / np.sqrt(
        summary["n_participants"]
    )
    return participant_topics, summary


def run_analysis(args: argparse.Namespace) -> None:
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install all_questions/requirements.txt."
        ) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = load_long_responses(
        args.conditions.expanduser().resolve(),
        args.interviews.expanduser().resolve(),
    )
    responses = prepare_responses(responses)
    responses.to_csv(output_dir / "combined_responses_audit.csv", index=False)
    analysis = responses.loc[responses["included_in_response_model"]].reset_index(
        drop=True
    )
    if len(analysis) < max(8, args.min_cluster_size * 2):
        raise ValueError("Too few eligible responses for BERTopic.")

    documents = analysis["response"].tolist()
    formatted = [f"{EMBED_INSTRUCTION}{document}" for document in documents]
    model_kwargs = {"device": args.device} if args.device else {}
    embedding_model = SentenceTransformer(MODEL_ID, **model_kwargs)
    embeddings = embedding_model.encode(
        formatted,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    umap_model = UMAP(
        n_neighbors=min(10, len(documents) - 1),
        n_components=min(5, len(documents) - 2),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method=args.cluster_selection_method,
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words=PORTUGUESE_STOP_WORDS,
        min_df=1,
        token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b",
    )
    topic_model = BERTopic(
        embedding_model=None,
        language="multilingual",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True,
    )
    topic_model.fit_transform(documents, embeddings)

    topic_info = topic_model.get_topic_info().copy()
    topic_info["Human_Label"] = topic_info["Topic"].map(response_topic_label)
    topic_info.to_csv(output_dir / "combined_responses_topics.csv", index=False)
    topic_info[["Topic", "Count", "Human_Label"]].to_csv(
        output_dir / "combined_responses_topic_labels.csv", index=False
    )
    document_info = topic_model.get_document_info(documents).reset_index(drop=True)
    document_output = pd.concat([analysis, document_info], axis=1)
    document_output["Human_Label"] = document_output["Topic"].map(response_topic_label)
    document_output.to_csv(
        output_dir / "combined_responses_with_topics.csv", index=False
    )
    topic_names = topic_info.set_index("Topic")["Name"].to_dict()

    raw = raw_response_prevalence(document_output, topic_names)
    raw.to_csv(
        output_dir / "combined_responses_topic_prevalence_by_condition.csv",
        index=False,
    )
    participant_topics, participant_summary = participant_response_prevalence(
        document_output, topic_names
    )
    participant_topics.to_csv(
        output_dir / "combined_responses_participant_topic_proportions.csv",
        index=False,
    )
    participant_summary.to_csv(
        output_dir
        / "combined_responses_topic_prevalence_by_condition_participant_weighted.csv",
        index=False,
    )

    terms_by_condition = topic_model.topics_per_class(
        documents, classes=analysis["condition_label"].tolist()
    )
    terms_by_condition["Human_Label"] = terms_by_condition["Topic"].map(
        response_topic_label
    )
    terms_by_condition.to_csv(
        output_dir / "combined_responses_topic_terms_by_condition.csv", index=False
    )
    terms_by_question = topic_model.topics_per_class(
        documents, classes=analysis["question_id"].tolist()
    )
    terms_by_question["Human_Label"] = terms_by_question["Topic"].map(
        response_topic_label
    )
    terms_by_question.to_csv(
        output_dir / "combined_responses_topic_terms_by_question.csv", index=False
    )
    if args.save_model:
        topic_model.save(
            output_dir / "combined_responses_bertopic_model",
            serialization="safetensors",
            save_ctfidf=True,
        )

    summary = {
        "analysis_unit": "One complete response to one interview question.",
        "corpus": "All non-empty responses to Q1, Q2, and Q3 pooled together.",
        "model_id": MODEL_ID,
        "embedding_instruction": EMBED_INSTRUCTION.strip(),
        "n_response_rows": int(len(responses)),
        "n_analyzed_responses": int(len(analysis)),
        "n_participants_analyzed": int(analysis["participant_id"].nunique()),
        "n_topics_excluding_outlier": int((topic_info["Topic"] != -1).sum()),
        "responses_by_question": analysis.groupby("question_id").size().to_dict(),
        "responses_by_condition": analysis.groupby("condition_label").size().to_dict(),
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_method": args.cluster_selection_method,
        "random_state": 42,
        "package_versions": {
            package: package_version(package)
            for package in [
                "bertopic",
                "sentence-transformers",
                "transformers",
                "umap-learn",
                "hdbscan",
                "pandas",
            ]
        },
    }
    (output_dir / "combined_responses_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Analyzed {len(analysis)} complete responses from "
        f"{analysis['participant_id'].nunique()} participants."
    )


if __name__ == "__main__":
    run_analysis(parse_args())
