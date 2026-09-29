"""Shared loading and helpers for the pre/post analyses."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "topic_network_post"))

from pagerank_within_condition_tests import counts_from, pagerank, transition_matrix  # noqa: E402,F401
from global_distribution_tests import gjsd_terms, holm  # noqa: E402,F401
from topic_network_analysis import bh_fdr  # noqa: E402,F401

OUT = HERE / "outputs"
CONDITIONS = ["VR Art", "VR Only", "Control"]
TIMES = ["pre", "post"]
COLORS = {"VR Art": "#8de5a1", "VR Only": "#ffb482", "Control": "#a1c9f4"}
DARK = {"VR Art": "#3fae5d", "VR Only": "#e0823a", "Control": "#5b8fd1"}

# Short labels: an interpretation written after reading sample units of every topic
# (pre_post/outputs/prepost_units_with_topics.csv); topics 2, 8 and 9 overlap in meaning.
TOPIC_LABELS = {
    -1: "Unassigned",
    0: "Anxiety & fear",
    1: "Feeling fine",
    2: "Calm & tranquil",
    3: "The surgery",
    4: "Mixed emotions",
    5: "Optimism",
    6: "Video & nature",
    7: "Expecting it to go well",
    8: "Feeling calmer",
    9: "Relaxed",
    10: "Wanting it over",
}


def label(t: int) -> str:
    return TOPIC_LABELS.get(int(t), f"Topic {int(t)}")


def load_units(drop_review: bool = False, drop_outlier: bool = True) -> pd.DataFrame:
    u = pd.read_csv(OUT / "prepost_units_with_topics.csv")
    if drop_review:
        u = u[~u.needs_review]
    if drop_outlier:
        u = u[u.Topic >= 0]
    return u.sort_values(["participant_id", "time", "sentence_number"]).reset_index(drop=True)


def embeddings(u: pd.DataFrame) -> np.ndarray:
    z = np.load(OUT / "unit_embeddings.npz")
    lk = dict(zip(z["document_id"].tolist(), z["emb"]))
    E = np.stack([lk[i] for i in u["document_id"]]).astype(np.float64)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def participants(u: pd.DataFrame) -> pd.DataFrame:
    p = u.drop_duplicates("participant_id")[["participant_id", "condition_label"]].copy()
    p["cond"] = p.condition_label.map(CONDITIONS.index)
    has = u.groupby("participant_id").time.agg(lambda s: set(s))
    p["has_pre"] = p.participant_id.map(lambda i: "pre" in has[i])
    p["has_post"] = p.participant_id.map(lambda i: "post" in has[i])
    return p.sort_values("participant_id").reset_index(drop=True)
