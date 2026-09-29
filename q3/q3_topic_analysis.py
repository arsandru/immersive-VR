#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_INSTRUCTION = (
    "Instruct: Represent the main topic and semantic meaning of this Portuguese "
    "interview sentence.\nText: "
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONDITIONS = BASE_DIR / "data" / "participant_conditions.csv"
DEFAULT_INTERVIEWS = BASE_DIR / "data" / "immerse_interviewPost.csv"
DEFAULT_OUTPUT = BASE_DIR / "q3" / "outputs"

PARTICIPANT_COLUMN = "Participant"
CONDITION_ID_COLUMN = "enrollment_number"
CONDITION_COLUMN = "condition_number"
QUESTION_COLUMN = "Que aspetos do cuidado prestado foram mais importantes para si?"

CONDITION_LABELS = {
    "1": "VR Art",
    "2": "VR Only",
    "3": "Control",
}

PORTUGUESE_STOP_WORDS = [
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "até", "com", "como", "da", "das", "de", "dela", "dele", "deles",
    "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
    "era", "essa", "essas", "esse", "esses", "esta", "está", "estão", "estas",
    "este", "estes", "eu", "foi", "foram", "há", "isso", "isto", "já", "lhe",
    "mais", "mas", "me", "mesmo", "meu", "minha", "muito", "muita", "muitos",
    "muitas", "na", "nas", "não", "nem", "no", "nos", "nós", "num", "numa",
    "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "porque",
    "qual", "quando", "que", "quem", "se", "sem", "ser", "seu", "sua", "são",
    "também", "tem", "têm", "ter", "toda", "todo", "todos", "tudo", "um", "uma",
    "umas", "uns", "você", "vocês",
    # Frequent discourse/evaluation terms that obscure care-related topic words.
    "algum", "acho", "aqui", "assim", "bem", "boa", "bom", "coisa", "coisas",
    "dizer", "então", "estava", "estar", "estou", "fui", "houve", "nada", "pois",
    "portanto", "princípio", "princpio", "pronto", "quer", "sempre", "sido",
    "sim", "simplesmente", "só", "tenho", "teve", "tive",
]

TOKEN_PATTERN = re.compile(r"(?u)\b[^\W\d_][^\W\d_]+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join Q3 interview responses to condition assignments, split them into "
            "sentences, embed with Qwen3-Embedding-0.6B, and fit BERTopic."
        )
    )
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--interviews", type=Path, default=DEFAULT_INTERVIEWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or mps.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--cluster-selection-method",
        choices=("leaf", "eom"),
        default="leaf",
        help="HDBSCAN cluster selection method. 'leaf' gives finer topics for this small corpus.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the fitted BERTopic model under the output directory.",
    )
    return parser.parse_args()


def canonical_column(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip().strip("*")
    value = re.sub(r"[_\s]+", " ", value)
    return value.casefold()


def resolve_column(columns: pd.Index, requested: str) -> str:
    canonical_requested = canonical_column(requested)
    matches = [column for column in columns if canonical_column(column) == canonical_requested]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one column matching {requested!r}; found {matches or 'none'}. "
            f"Available columns: {list(columns)}"
        )
    return matches[0]


def normalize_identifier(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.str.replace(r"\.0$", "", regex=True)


def clean_response(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def split_sentences(text: str) -> list[str]:
    if not text:
        return []

    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    # Handle transcript joins such as "excelente.Foram" before splitting.
    normalized = re.sub(r"(?<=[.!?])(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])", "\n", normalized)
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def content_tokens(text: str) -> list[str]:
    stop_words = set(PORTUGUESE_STOP_WORDS)
    return [
        token
        for token in TOKEN_PATTERN.findall(text.casefold())
        if token not in stop_words
    ]


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def load_joined_data(
    conditions_path: Path,
    interviews_path: Path,
) -> tuple[pd.DataFrame, str]:
    conditions = pd.read_csv(conditions_path, dtype=str, encoding="utf-8-sig")
    interviews = pd.read_csv(interviews_path, dtype=str, encoding="utf-8-sig")

    enrollment_col = resolve_column(conditions.columns, CONDITION_ID_COLUMN)
    condition_col = resolve_column(conditions.columns, CONDITION_COLUMN)
    participant_col = resolve_column(interviews.columns, PARTICIPANT_COLUMN)
    question_col = resolve_column(interviews.columns, QUESTION_COLUMN)

    conditions = conditions.copy()
    interviews = interviews.copy()
    conditions["participant_id"] = normalize_identifier(conditions[enrollment_col])
    conditions[condition_col] = normalize_identifier(conditions[condition_col])
    interviews["participant_id"] = normalize_identifier(interviews[participant_col])

    duplicate_conditions = conditions.loc[
        conditions["participant_id"].duplicated(keep=False), "participant_id"
    ].unique()
    if len(duplicate_conditions):
        raise ValueError(
            "Condition mapping contains duplicate enrollment numbers: "
            + ", ".join(map(str, duplicate_conditions))
        )

    condition_map = conditions[["participant_id", condition_col]].rename(
        columns={condition_col: "condition_number"}
    )
    merged = interviews.merge(
        condition_map,
        on="participant_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    merged.insert(0, "response_id", np.arange(1, len(merged) + 1))
    merged["response"] = clean_response(merged[question_col])
    merged["has_response"] = merged["response"].ne("")
    merged["has_condition"] = merged["_merge"].eq("both")
    merged["included_in_model"] = merged["has_response"] & merged["has_condition"]
    mapped_labels = merged["condition_number"].map(CONDITION_LABELS)
    fallback_labels = "Condition " + merged["condition_number"].fillna("unmatched")
    merged["condition_label"] = mapped_labels.fillna(fallback_labels)
    merged["exclusion_reason"] = np.select(
        [~merged["has_response"], ~merged["has_condition"]],
        ["blank response", "condition not matched"],
        default="",
    )
    return merged.drop(columns="_merge"), question_col


def build_sentence_data(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    document_id = 1

    for row in joined.itertuples(index=False):
        if not row.has_response:
            continue

        for sentence_number, sentence in enumerate(split_sentences(row.response), start=1):
            tokens = content_tokens(sentence)
            has_content = bool(tokens)
            included = bool(row.has_condition and has_content)
            if not row.has_condition:
                exclusion_reason = "condition not matched"
            elif not has_content:
                exclusion_reason = "no substantive tokens after stopword removal"
            else:
                exclusion_reason = ""

            rows.append(
                {
                    "document_id": document_id,
                    "response_id": row.response_id,
                    "participant_id": row.participant_id,
                    "sentence_number": sentence_number,
                    "condition_number": row.condition_number,
                    "condition_label": row.condition_label,
                    "response": row.response,
                    "sentence": sentence,
                    "content_text": " ".join(tokens),
                    "content_token_count": len(tokens),
                    "has_condition": row.has_condition,
                    "included_in_model": included,
                    "exclusion_reason": exclusion_reason,
                }
            )
            document_id += 1

    return pd.DataFrame(rows)


def participant_weighted_prevalence(
    document_output: pd.DataFrame,
    topic_names: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = document_output[
        ["participant_id", "condition_number", "condition_label"]
    ].drop_duplicates()
    topics = pd.DataFrame({"Topic": sorted(document_output["Topic"].unique())})
    complete_grid = participants.merge(topics, how="cross")

    counts = (
        document_output.groupby(
            ["participant_id", "condition_number", "condition_label", "Topic"]
        )
        .size()
        .rename("n_sentences")
        .reset_index()
    )
    totals = (
        document_output.groupby("participant_id")
        .size()
        .rename("n_participant_sentences")
        .reset_index()
    )
    participant_topics = complete_grid.merge(
        counts,
        on=["participant_id", "condition_number", "condition_label", "Topic"],
        how="left",
    ).merge(totals, on="participant_id", how="left")
    participant_topics["n_sentences"] = participant_topics["n_sentences"].fillna(0).astype(int)
    participant_topics["participant_topic_proportion"] = (
        participant_topics["n_sentences"] / participant_topics["n_participant_sentences"]
    )
    participant_topics["topic_name"] = participant_topics["Topic"].map(topic_names)

    condition_summary = (
        participant_topics.groupby(
            ["condition_number", "condition_label", "Topic", "topic_name"],
            dropna=False,
        )["participant_topic_proportion"]
        .agg(n_participants="size", mean_proportion="mean", sd_proportion="std")
        .reset_index()
    )
    condition_summary["se_proportion"] = (
        condition_summary["sd_proportion"] / np.sqrt(condition_summary["n_participants"])
    )
    return participant_topics, condition_summary


def run_analysis(args: argparse.Namespace) -> None:
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise SystemExit(
            "Missing q3 dependencies. Install them with: "
            "python -m pip install -r q3/requirements.txt"
        ) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    joined, source_question_col = load_joined_data(
        args.conditions.expanduser().resolve(),
        args.interviews.expanduser().resolve(),
    )
    response_audit_columns = [
        "response_id",
        "participant_id",
        "condition_number",
        "condition_label",
        "response",
        "has_response",
        "has_condition",
        "included_in_model",
        "exclusion_reason",
    ]
    joined[response_audit_columns].to_csv(output_dir / "q3_join_audit.csv", index=False)

    sentences = build_sentence_data(joined)
    sentences.to_csv(output_dir / "q3_sentence_audit.csv", index=False)
    analysis = sentences.loc[sentences["included_in_model"]].reset_index(drop=True)
    if len(analysis) < max(8, args.min_cluster_size * 2):
        raise ValueError(
            f"Only {len(analysis)} eligible sentences are available; "
            "this is too few for the requested clustering settings."
        )

    documents = analysis["sentence"].tolist()
    formatted_documents = [f"{EMBED_INSTRUCTION}{document}" for document in documents]

    model_kwargs = {"device": args.device} if args.device else {}
    embedding_model = SentenceTransformer(MODEL_ID, **model_kwargs)
    embeddings = embedding_model.encode(
        formatted_documents,
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
    topic_info.to_csv(output_dir / "q3_topics.csv", index=False)

    document_info = topic_model.get_document_info(documents).reset_index(drop=True)
    document_output = pd.concat([analysis.reset_index(drop=True), document_info], axis=1)
    document_output.to_csv(output_dir / "q3_documents_with_topics.csv", index=False)

    topic_names = topic_info.set_index("Topic")["Name"].to_dict()
    prevalence = (
        document_output.groupby(
            ["condition_number", "condition_label", "Topic"],
            dropna=False,
        )
        .size()
        .rename("n_sentences")
        .reset_index()
    )
    prevalence["n_condition_sentences"] = prevalence.groupby("condition_number")[
        "n_sentences"
    ].transform("sum")
    prevalence["proportion_within_condition"] = (
        prevalence["n_sentences"] / prevalence["n_condition_sentences"]
    )
    prevalence["topic_name"] = prevalence["Topic"].map(topic_names)
    prevalence = prevalence[
        [
            "condition_number",
            "condition_label",
            "Topic",
            "topic_name",
            "n_sentences",
            "n_condition_sentences",
            "proportion_within_condition",
        ]
    ].sort_values(["condition_number", "Topic"])
    prevalence.to_csv(output_dir / "q3_topic_prevalence_by_condition.csv", index=False)

    participant_topics, participant_condition_summary = participant_weighted_prevalence(
        document_output,
        topic_names,
    )
    participant_topics.to_csv(
        output_dir / "q3_participant_topic_proportions.csv",
        index=False,
    )
    participant_condition_summary.to_csv(
        output_dir / "q3_topic_prevalence_by_condition_participant_weighted.csv",
        index=False,
    )

    topics_by_condition = topic_model.topics_per_class(
        documents,
        classes=analysis["condition_label"].tolist(),
    )
    topics_by_condition.to_csv(output_dir / "q3_topic_terms_by_condition.csv", index=False)

    if args.save_model:
        topic_model.save(
            output_dir / "bertopic_model",
            serialization="safetensors",
            save_ctfidf=True,
        )

    condition_counts = (
        analysis.groupby(["condition_number", "condition_label"])
        .agg(
            n_sentences=("document_id", "size"),
            n_participants=("participant_id", "nunique"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "model_id": MODEL_ID,
        "embedding_instruction": EMBED_INSTRUCTION.strip(),
        "embedding_text": "Original sentence with stopwords retained for semantic context.",
        "topic_representation_text": "Stopwords removed by CountVectorizer.",
        "source_question_column": source_question_col,
        "n_interview_rows": int(len(joined)),
        "n_responses_with_text": int(joined["has_response"].sum()),
        "n_sentence_rows": int(len(sentences)),
        "n_analyzed_sentences": int(len(analysis)),
        "n_participants_analyzed": int(analysis["participant_id"].nunique()),
        "n_blank_responses": int((~joined["has_response"]).sum()),
        "n_unmatched_participants": int((~joined["has_condition"]).sum()),
        "n_sentences_without_substantive_tokens": int(
            (sentences["content_token_count"] == 0).sum()
        ),
        "unmatched_participant_ids": joined.loc[
            ~joined["has_condition"], "participant_id"
        ].tolist(),
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
    (output_dir / "q3_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Analyzed {len(analysis)} sentences from "
        f"{analysis['participant_id'].nunique()} participants across "
        f"{len(condition_counts)} conditions."
    )
    print(
        f"Excluded {(~sentences['included_in_model']).sum()} sentence rows; "
        "see q3_sentence_audit.csv."
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    run_analysis(parse_args())
