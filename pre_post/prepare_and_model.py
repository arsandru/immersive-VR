#!/usr/bin/env python3
"""Pre/post Q1 descriptors -> one shared embedding + BERTopic model.

Units are the rater-consensus descriptors (split on ';', order kept) of Q1
("how do you feel now?") from data/consensus_pre_q1.csv and consensus_post_q1.csv.
Pre and post are pooled so that topics mean the same at both timepoints.
Participant ids are enrollment numbers (data/participant_conditions.csv).

Data handling (all logged in outputs/prepost_audit.json):
* descriptors are split on ';' and stripped; exact case-insensitive repeats within
  one participant/timepoint are dropped;
* participants without a condition are excluded;
* rows flagged needs_review are kept but marked (sensitivity analyses can drop them).
"""
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
    EMBED_INSTRUCTION, MODEL_ID, PORTUGUESE_STOP_WORDS, clean_response,
    content_tokens,
)

DATA = BASE_DIR / "data"
COND_NAMES = {1: "VR Art", 2: "VR Only", 3: "Control"}  # same coding as the post-interview analyses
OUT = Path(__file__).resolve().parent / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--cluster-selection-method", choices=("leaf", "eom"), default="eom")
    return p.parse_args()


def load_units(audit: dict) -> pd.DataFrame:
    cond = pd.read_csv(DATA / "participant_conditions.csv")
    cmap = cond.set_index("enrollment_number")["condition_number"].to_dict()
    rows = []
    for time in ("pre", "post"):
        d = pd.read_csv(DATA / f"consensus_{time}_q1.csv")
        audit[f"{time}_participants_in_file"] = int(d.participant_id.nunique())
        audit[f"{time}_missing_descriptors"] = d.loc[d.descriptors.isna(), "participant_id"].tolist()
        audit[f"{time}_no_condition"] = sorted(set(d.participant_id) - set(cmap))
        d = d.dropna(subset=["descriptors"])
        d = d[d.participant_id.isin(cmap)]
        dup = 0
        for r in d.itertuples():
            seen = set()
            n = 0
            for part in str(r.descriptors).split(";"):
                text = clean_response(pd.Series([part])).iloc[0]
                if not text:
                    continue
                if text.casefold() in seen:
                    dup += 1
                    continue
                seen.add(text.casefold())
                n += 1
                rows.append({"participant_id": int(r.participant_id), "time": time,
                             "condition_number": int(cmap[r.participant_id]),
                             "condition_label": COND_NAMES[int(cmap[r.participant_id])],
                             "sentence_number": n, "sentence": text,
                             "needs_review": bool(r.needs_review)})
        audit[f"{time}_duplicates_dropped"] = dup
    df = pd.DataFrame(rows)
    df["response_id"] = df["participant_id"].astype(str) + "_" + df["time"]
    df["question_id"] = "Q1"
    df["content_text"] = df["sentence"].map(lambda t: " ".join(content_tokens(t)))
    df["included_in_model"] = True
    df.insert(0, "document_id", np.arange(1, len(df) + 1))
    audit["units"] = df.groupby("time").size().to_dict()
    audit["participants"] = df.groupby("time").participant_id.nunique().to_dict()
    audit["participants_both"] = len(set(df[df.time == "pre"].participant_id) & set(df[df.time == "post"].participant_id))
    return df


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit: dict = {}
    df = load_units(audit)
    print(json.dumps(audit, indent=1, default=str))

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    kw = {"device": args.device} if args.device else {}
    model = SentenceTransformer(MODEL_ID, **kw)
    emb = model.encode([f"{EMBED_INSTRUCTION}{s}" for s in df["sentence"]], batch_size=args.batch_size,
                       show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    np.savez(out / "unit_embeddings.npz", document_id=df["document_id"].to_numpy(), emb=emb)

    docs = df["sentence"].tolist()
    umap_model = UMAP(n_neighbors=min(10, len(docs) - 1), n_components=5, min_dist=0.0,
                      metric="cosine", random_state=42)
    hdb = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                  metric="euclidean", cluster_selection_method=args.cluster_selection_method,
                  prediction_data=True)
    vec = CountVectorizer(lowercase=True, ngram_range=(1, 2), stop_words=PORTUGUESE_STOP_WORDS, min_df=1,
                          token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")
    tm = BERTopic(embedding_model=None, language="multilingual", umap_model=umap_model, hdbscan_model=hdb,
                  vectorizer_model=vec, calculate_probabilities=True, verbose=False)
    tm.fit_transform(docs, emb)
    info = tm.get_topic_info()
    info.to_csv(out / "prepost_topics.csv", index=False)
    di = tm.get_document_info(docs).reset_index(drop=True)
    res = pd.concat([df, di[["Topic", "Name", "Probability", "Representative_document"]]], axis=1)
    res.to_csv(out / "prepost_units_with_topics.csv", index=False)
    audit["n_topics_excl_outlier"] = int((info.Topic >= 0).sum())
    audit["outlier_share"] = float((res.Topic == -1).mean())
    audit["bertopic"] = {"min_cluster_size": args.min_cluster_size, "min_samples": args.min_samples,
                         "cluster_selection_method": args.cluster_selection_method}
    (out / "prepost_audit.json").write_text(json.dumps(audit, indent=1, default=str))
    pd.set_option("display.width", 250, "display.max_colwidth", 110)
    print(info[["Topic", "Count", "Name"]].to_string(index=False))
    print("outlier share:", round(audit["outlier_share"], 3))


if __name__ == "__main__":
    main()
