#!/usr/bin/env python3
"""Prepare the pre-interview descriptors for the topic-network scripts (same input format as the post analysis).

Takes the pre units of pre_post/outputs/prepost_units_with_topics.csv (one shared BERTopic model fitted on pre + post
descriptors, so topics mean the same at both timepoints) and writes
  outputs/pre_descriptor_topic_assignments.csv  one row per pre descriptor, in the column layout of the post analysis
  outputs/sentence_embeddings.npz               cached Qwen3 embeddings of those descriptors (from pre_post)
Every participant has one 'response' (their pre Q1 list), so the within-response and across-question flow scopes of the
post analysis coincide; only within_response is used here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PP = HERE.parent / "pre_post" / "outputs"
u = pd.read_csv(PP / "prepost_units_with_topics.csv")
u = u[u.time == "pre"].copy()
u["response_id"] = u["participant_id"]
u["question_id"] = "Q1"
u["included_in_model"] = True
u.to_csv(HERE / "outputs" / "pre_descriptor_topic_assignments.csv", index=False)
z = np.load(PP / "unit_embeddings.npz")
lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
ids = u["document_id"].to_numpy()
np.savez(HERE / "outputs" / "sentence_embeddings.npz", document_id=ids, emb=np.stack([lookup[i] for i in ids]))
print(len(u), "pre descriptors,", u.participant_id.nunique(), "participants,", u.Topic.nunique(), "topics incl. outlier")
