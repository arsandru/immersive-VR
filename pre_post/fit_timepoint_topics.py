#!/usr/bin/env python3
"""Fit a SEPARATE BERTopic model on the Q1 descriptors of one timepoint (pre or post), so that a timepoint's topics are
extracted from its own descriptors only (no information from the other interview).

Uses the cleaned units and cached Qwen3 embeddings of prepare_and_model.py (an embedding depends only on its own text).
Same UMAP / HDBSCAN / c-TF-IDF settings as the pooled model except min_cluster_size (smaller samples).
Outputs: outputs/{time}_units_with_topics.csv, outputs/{time}_topics.csv, outputs/{time}_topic_audit.json
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
from q3.q3_topic_analysis import PORTUGUESE_STOP_WORDS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from timepoint_topic_labels import TOPIC_MODEL_PARAMS as P  # noqa: E402  (single source of the parameters, same for pre and post)

OUT = Path(__file__).resolve().parent / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time", choices=("pre", "post"), required=True)
    p.add_argument("--min-cluster-size", type=int, default=P["min_cluster_size"])
    p.add_argument("--min-samples", type=int, default=P["min_samples"])
    p.add_argument("--tag", default="", help="suffix for the output files (for exploration)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    u = pd.read_csv(OUT / "prepost_units_with_topics.csv")
    u = u[u.time == a.time].drop(columns=[c for c in ("Topic", "Name", "Probability", "Representative_document") if c in u.columns])
    u = u.reset_index(drop=True)
    z = np.load(OUT / "unit_embeddings.npz")
    lk = dict(zip(z["document_id"].tolist(), z["emb"]))
    emb = np.stack([lk[i] for i in u["document_id"]])
    docs = u["sentence"].tolist()
    umap_model = UMAP(n_neighbors=min(P["umap"]["n_neighbors"], len(docs) - 1), n_components=P["umap"]["n_components"],
                      min_dist=P["umap"]["min_dist"], metric=P["umap"]["metric"], random_state=P["umap"]["random_state"])
    hdb = HDBSCAN(min_cluster_size=a.min_cluster_size, min_samples=a.min_samples, metric="euclidean",
                  cluster_selection_method=P["cluster_selection_method"], prediction_data=True)
    vec = CountVectorizer(lowercase=True, ngram_range=(1, 2), stop_words=PORTUGUESE_STOP_WORDS, min_df=1,
                          token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")
    tm = BERTopic(embedding_model=None, language="multilingual", umap_model=umap_model, hdbscan_model=hdb,
                  vectorizer_model=vec, calculate_probabilities=True, verbose=False)
    tm.fit_transform(docs, emb)
    info = tm.get_topic_info()
    di = tm.get_document_info(docs).reset_index(drop=True)
    res = pd.concat([u, di[["Topic", "Name", "Probability", "Representative_document"]]], axis=1)
    res.to_csv(OUT / f"{a.time}_units_with_topics{a.tag}.csv", index=False)
    info.to_csv(OUT / f"{a.time}_topics{a.tag}.csv", index=False)
    audit = {"time": a.time, "n_units": len(u), "min_cluster_size": a.min_cluster_size, "min_samples": a.min_samples,
             "n_topics_excl_outlier": int((info.Topic >= 0).sum()), "outlier_share": float((res.Topic == -1).mean()),
             "topic_sizes": res.Topic.value_counts().sort_index().to_dict()}
    (OUT / f"{a.time}_topic_audit{a.tag}.json").write_text(json.dumps(audit, indent=1, default=str))
    print(json.dumps(audit, default=str))


if __name__ == "__main__":
    main()
