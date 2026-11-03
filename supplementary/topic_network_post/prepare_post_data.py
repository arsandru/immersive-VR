#!/usr/bin/env python3
"""Prepare the post-interview Q1 descriptors for the topic-network scripts.

Takes pre_post/outputs/post_units_with_topics.csv: the clusters of the BERTopic model fitted on the POST descriptors only
(pre_post/fit_timepoint_topics.py; same parameters and label source as topic_network_pre/ and the core networks of pre_post/),
and writes
  outputs/post_descriptor_topic_assignments.csv  one row per post descriptor, in the column layout the scripts expect
  outputs/sentence_embeddings.npz                cached Qwen3 embeddings of those descriptors (from pre_post)
Every participant has one 'response' (their post Q1 list), so the within-response and across-question flow scopes coincide;
only within_response is used. (The earlier analysis of the full post interview, three questions, is kept in
topic_network_post_allquestions/.)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PP = HERE.parent / "pre_post" / "outputs"
u = pd.read_csv(PP / "post_units_with_topics.csv")
u["response_id"] = u["participant_id"]
u["question_id"] = "Q1"
u["included_in_model"] = True
u.to_csv(HERE / "outputs" / "post_descriptor_topic_assignments.csv", index=False)
z = np.load(PP / "unit_embeddings.npz")
lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
ids = u["document_id"].to_numpy()
np.savez(HERE / "outputs" / "sentence_embeddings.npz", document_id=ids, emb=np.stack([lookup[i] for i in ids]))
print(len(u), "post descriptors,", u.participant_id.nunique(), "participants,", u.Topic.nunique(), "topics incl. outlier")
