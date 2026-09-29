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

from q3.q3_topic_analysis import (  # noqa: E402
    CONDITION_COLUMN,
    CONDITION_ID_COLUMN,
    CONDITION_LABELS,
    EMBED_INSTRUCTION,
    MODEL_ID,
    PARTICIPANT_COLUMN,
    PORTUGUESE_STOP_WORDS,
    clean_response,
    content_tokens,
    normalize_identifier,
    package_version,
    resolve_column,
    split_sentences,
)
from all_questions.topic_labels import topic_label  # noqa: E402


DEFAULT_CONDITIONS = BASE_DIR / "data" / "participant_conditions.csv"
DEFAULT_INTERVIEWS = BASE_DIR / "data" / "immerse_interviewPost.csv"
DEFAULT_OUTPUT = BASE_DIR / "all_questions" / "outputs"

QUESTION_COLUMNS = {
    "Q1": "Como se sente agora?",
    "Q2": "Sente alguma diferença em relação a como se sentia antes? ",
    "Q3": "Que aspetos do cuidado prestado foram mais importantes para si?",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stack all three post-interview questions, split responses into sentences, "
            "embed them with Qwen3-Embedding-0.6B, and fit one shared BERTopic model."
        )
    )
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--interviews", type=Path, default=DEFAULT_INTERVIEWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or mps.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--cluster-selection-method",
        choices=("leaf", "eom"),
        default="eom",
    )
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def load_long_responses(
    conditions_path: Path,
    interviews_path: Path,
) -> pd.DataFrame:
    conditions = pd.read_csv(conditions_path, dtype=str, encoding="utf-8-sig")
    interviews = pd.read_csv(interviews_path, dtype=str, encoding="utf-8-sig")

    enrollment_col = resolve_column(conditions.columns, CONDITION_ID_COLUMN)
    condition_col = resolve_column(conditions.columns, CONDITION_COLUMN)
    participant_col = resolve_column(interviews.columns, PARTICIPANT_COLUMN)
    resolved_questions = {
        question_id: resolve_column(interviews.columns, question)
        for question_id, question in QUESTION_COLUMNS.items()
    }

    conditions = conditions.copy()
    interviews = interviews.copy()
    conditions["participant_id"] = normalize_identifier(conditions[enrollment_col])
    conditions[condition_col] = normalize_identifier(conditions[condition_col])
    interviews["participant_id"] = normalize_identifier(interviews[participant_col])

    duplicated = conditions.loc[
        conditions["participant_id"].duplicated(keep=False), "participant_id"
    ].unique()
    if len(duplicated):
        raise ValueError(
            "Condition mapping contains duplicate enrollment numbers: "
            + ", ".join(map(str, duplicated))
        )

    condition_map = conditions[["participant_id", condition_col]].rename(
        columns={condition_col: "condition_number"}
    )
    response_frames = []
    for question_id, source_column in resolved_questions.items():
        frame = interviews[["participant_id", source_column]].copy()
        frame["question_id"] = question_id
        frame["question_text"] = QUESTION_COLUMNS[question_id].strip()
        frame["response"] = clean_response(frame[source_column])
        response_frames.append(
            frame[["participant_id", "question_id", "question_text", "response"]]
        )

    long_responses = pd.concat(response_frames, ignore_index=True)
    merged = long_responses.merge(
        condition_map,
        on="participant_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    merged.insert(0, "response_id", np.arange(1, len(merged) + 1))
    merged["has_response"] = merged["response"].ne("")
    merged["has_condition"] = merged["_merge"].eq("both")
    merged["included_in_model"] = merged["has_response"] & merged["has_condition"]
    mapped = merged["condition_number"].map(CONDITION_LABELS)
    fallback = "Condition " + merged["condition_number"].fillna("unmatched")
    merged["condition_label"] = mapped.fillna(fallback)
    merged["exclusion_reason"] = np.select(
        [~merged["has_response"], ~merged["has_condition"]],
        ["blank response", "condition not matched"],
        default="",
    )
    return merged.drop(columns="_merge")


def build_sentence_data(responses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    document_id = 1
    for row in responses.itertuples(index=False):
        if not row.has_response:
            continue
        for sentence_number, sentence in enumerate(split_sentences(row.response), start=1):
            tokens = content_tokens(sentence)
            has_content = bool(tokens)
            included = bool(row.has_condition and has_content)
            if not row.has_condition:
                reason = "condition not matched"
            elif not has_content:
                reason = "no substantive tokens after stopword removal"
            else:
                reason = ""
            rows.append(
                {
                    "document_id": document_id,
                    "response_id": row.response_id,
                    "participant_id": row.participant_id,
                    "question_id": row.question_id,
                    "question_text": row.question_text,
                    "sentence_number": sentence_number,
                    "condition_number": row.condition_number,
                    "condition_label": row.condition_label,
                    "response": row.response,
                    "sentence": sentence,
                    "content_text": " ".join(tokens),
                    "content_token_count": len(tokens),
                    "has_condition": row.has_condition,
                    "included_in_model": included,
                    "exclusion_reason": reason,
                }
            )
            document_id += 1
    return pd.DataFrame(rows)


def raw_prevalence(
    documents: pd.DataFrame,
    topic_names: dict[int, str],
    grouping: list[str],
) -> pd.DataFrame:
    result = documents.groupby(grouping + ["Topic"], dropna=False).size().rename(
        "n_sentences"
    ).reset_index()
    result["n_group_sentences"] = result.groupby(grouping)["n_sentences"].transform("sum")
    result["proportion_within_group"] = result["n_sentences"] / result["n_group_sentences"]
    result["topic_name"] = result["Topic"].map(topic_names)
    result["topic_label"] = result["Topic"].map(topic_label)
    return result.sort_values(grouping + ["Topic"])


def participant_prevalence(
    documents: pd.DataFrame,
    topic_names: dict[int, str],
    unit_columns: list[str],
    summary_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity_columns = unit_columns + ["condition_number", "condition_label"]
    units = documents[identity_columns].drop_duplicates()
    topics = pd.DataFrame({"Topic": sorted(documents["Topic"].unique())})
    grid = units.merge(topics, how="cross")

    counts = documents.groupby(identity_columns + ["Topic"]).size().rename(
        "n_sentences"
    ).reset_index()
    totals = documents.groupby(unit_columns).size().rename("n_unit_sentences").reset_index()
    participant_topics = grid.merge(
        counts,
        on=identity_columns + ["Topic"],
        how="left",
    ).merge(totals, on=unit_columns, how="left")
    participant_topics["n_sentences"] = participant_topics["n_sentences"].fillna(0).astype(int)
    participant_topics["participant_topic_proportion"] = (
        participant_topics["n_sentences"] / participant_topics["n_unit_sentences"]
    )
    participant_topics["topic_name"] = participant_topics["Topic"].map(topic_names)
    participant_topics["topic_label"] = participant_topics["Topic"].map(topic_label)

    summary = participant_topics.groupby(
        summary_columns + ["condition_number", "condition_label", "Topic", "topic_name", "topic_label"],
        dropna=False,
    )["participant_topic_proportion"].agg(
        n_participants="size",
        mean_proportion="mean",
        sd_proportion="std",
    ).reset_index()
    summary["se_proportion"] = summary["sd_proportion"] / np.sqrt(summary["n_participants"])
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
            "Missing dependencies. Install them with: "
            "python -m pip install -r all_questions/requirements.txt"
        ) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = load_long_responses(
        args.conditions.expanduser().resolve(),
        args.interviews.expanduser().resolve(),
    )
    responses.to_csv(output_dir / "all_questions_join_audit.csv", index=False)

    sentences = build_sentence_data(responses)
    sentences.to_csv(output_dir / "all_questions_sentence_audit.csv", index=False)
    analysis = sentences.loc[sentences["included_in_model"]].reset_index(drop=True)
    if len(analysis) < max(8, args.min_cluster_size * 2):
        raise ValueError("Too few eligible sentences for the requested clustering settings.")

    documents = analysis["sentence"].tolist()
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
    topic_info["Human_Label"] = topic_info["Topic"].map(topic_label)
    topic_info.to_csv(output_dir / "all_questions_topics.csv", index=False)
    topic_info[["Topic", "Count", "Human_Label"]].to_csv(
        output_dir / "all_questions_topic_labels.csv", index=False
    )
    document_info = topic_model.get_document_info(documents).reset_index(drop=True)
    document_output = pd.concat([analysis, document_info], axis=1)
    document_output["Human_Label"] = document_output["Topic"].map(topic_label)
    document_output.to_csv(output_dir / "all_questions_documents_with_topics.csv", index=False)
    document_output.to_csv(
        output_dir / "combined_questions_sentence_topic_assignments.csv", index=False
    )
    topic_names = topic_info.set_index("Topic")["Name"].to_dict()

    combined_sentence_prevalence = raw_prevalence(
        document_output,
        topic_names,
        ["condition_number", "condition_label"],
    )
    combined_sentence_prevalence.to_csv(
        output_dir / "all_questions_topic_prevalence_by_condition.csv", index=False
    )
    combined_sentence_prevalence.to_csv(
        output_dir / "combined_questions_sentence_topic_prevalence_by_condition.csv",
        index=False,
    )
    raw_prevalence(
        document_output,
        topic_names,
        ["question_id", "question_text", "condition_number", "condition_label"],
    ).to_csv(
        output_dir / "all_questions_topic_prevalence_by_question_condition.csv",
        index=False,
    )

    participant_topics, condition_summary = participant_prevalence(
        document_output,
        topic_names,
        unit_columns=["participant_id"],
        summary_columns=[],
    )
    participant_topics.to_csv(
        output_dir / "all_questions_participant_topic_proportions.csv", index=False
    )
    condition_summary.to_csv(
        output_dir / "all_questions_topic_prevalence_by_condition_participant_weighted.csv",
        index=False,
    )

    participant_question_topics, question_condition_summary = participant_prevalence(
        document_output,
        topic_names,
        unit_columns=["participant_id", "question_id", "question_text"],
        summary_columns=["question_id", "question_text"],
    )
    participant_question_topics.to_csv(
        output_dir / "all_questions_participant_question_topic_proportions.csv",
        index=False,
    )
    question_condition_summary.to_csv(
        output_dir
        / "all_questions_topic_prevalence_by_question_condition_participant_weighted.csv",
        index=False,
    )

    terms_by_condition = topic_model.topics_per_class(
        documents, classes=analysis["condition_label"].tolist()
    )
    terms_by_condition["Human_Label"] = terms_by_condition["Topic"].map(topic_label)
    terms_by_condition.to_csv(
        output_dir / "all_questions_topic_terms_by_condition.csv", index=False
    )
    terms_by_question = topic_model.topics_per_class(
        documents, classes=analysis["question_id"].tolist()
    )
    terms_by_question["Human_Label"] = terms_by_question["Topic"].map(topic_label)
    terms_by_question.to_csv(
        output_dir / "all_questions_topic_terms_by_question.csv", index=False
    )

    if args.save_model:
        topic_model.save(
            output_dir / "bertopic_model",
            serialization="safetensors",
            save_ctfidf=True,
        )

    question_counts = analysis.groupby("question_id").agg(
        n_sentences=("document_id", "size"),
        n_participants=("participant_id", "nunique"),
    ).reset_index().to_dict(orient="records")
    condition_counts = analysis.groupby(["condition_number", "condition_label"]).agg(
        n_sentences=("document_id", "size"),
        n_participants=("participant_id", "nunique"),
    ).reset_index().to_dict(orient="records")
    summary = {
        "model_id": MODEL_ID,
        "embedding_instruction": EMBED_INSTRUCTION.strip(),
        "question_columns": {key: value.strip() for key, value in QUESTION_COLUMNS.items()},
        "topic_model_scope": "One shared BERTopic model across all three questions.",
        "n_interview_rows": int(responses["participant_id"].nunique()),
        "n_response_rows": int(len(responses)),
        "n_responses_with_text": int(responses["has_response"].sum()),
        "n_sentence_rows": int(len(sentences)),
        "n_analyzed_sentences": int(len(analysis)),
        "n_participants_analyzed": int(analysis["participant_id"].nunique()),
        "n_topics_excluding_outlier": int((topic_info["Topic"] != -1).sum()),
        "question_counts": question_counts,
        "condition_counts": condition_counts,
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
    (output_dir / "all_questions_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Analyzed {len(analysis)} sentences from "
        f"{analysis['participant_id'].nunique()} participants across all three questions."
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    run_analysis(parse_args())
