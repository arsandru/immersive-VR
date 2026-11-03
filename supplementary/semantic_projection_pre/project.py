#!/usr/bin/env python3
"""Semantic projection of the Q1 descriptors of ONE timepoint (pre or post; detected from the folder name).

Same method as semantic_projection/semantic_projection.py: Qwen3-Embedding-0.6B with the emotional-valence
instruction, axis = mean of all (relief anchor - distress anchor) pairs, score = projection on the axis
(positive = relief). The axis uses Portuguese translations of the original anchor words (the descriptors are
Portuguese). The y coordinate of the figure is PC1 of the residual
(orthogonal) variance of THIS timepoint's descriptors only.

Data: data/consensus_{time}_q1.csv, cleaned as in pre_post/prepare_and_model.py.
Outputs: projection_descriptors.csv, projection_audit.json, probe_words.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
TIME = HERE.name.split("_")[-1]  # semantic_projection_pre -> "pre"
assert TIME in ("pre", "post"), HERE.name
sys.path.insert(0, str(HERE.parent / "pre_post"))
from prepare_and_model import load_units  # noqa: E402

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_INSTRUCTION = "Instruct: Represent the emotional valence and intensity of this word or short phrase.\nText: "
DISTRESS_PT = ["angustiado", "perigo", "apreensivo", "angústia", "tormento", "agitado"]
RELIEF_PT = ["aliviado", "confortado", "segurança", "acalmado", "tranquilizado", "à vontade"]
PROBES_PT = {"ansiosa": -1, "com medo": -1, "preocupada": -1, "muito nervosa": -1, "a ansiedade aumenta": -1,
             "calma": 1, "relaxada": 1, "tranquila": 1, "aliviada": 1, "sinto-me segura": 1}

model = SentenceTransformer(MODEL_ID)


def enc(texts):
    return model.encode([f"{EMBED_INSTRUCTION}{t}" for t in texts], convert_to_numpy=True, show_progress_bar=False)


def make_axis(pos, neg):
    p, n = enc(pos), enc(neg)
    return np.array([a - b for a in p for b in n]).mean(axis=0)


def main() -> None:
    audit: dict = {"time": TIME}
    u = load_units(audit)
    u = u[u.time == TIME].reset_index(drop=True)
    audit["descriptors"] = len(u)
    audit["participants"] = int(u.participant_id.nunique())
    audit["by_condition"] = u.groupby("condition_label").participant_id.nunique().to_dict()
    E = enc(u["sentence"].tolist())
    ax = make_axis(RELIEF_PT, DISTRESS_PT)
    out = u[["document_id", "participant_id", "condition_label", "time", "sentence_number", "sentence", "needs_review"]].copy()
    out["projection"] = E @ ax / np.linalg.norm(ax)
    unit = ax / np.linalg.norm(ax)
    out["y_pc1_residual"] = PCA(n_components=1, random_state=0).fit_transform(E - np.outer(E @ unit, unit)).ravel()
    out.to_csv(HERE / "projection_descriptors.csv", index=False)
    pe = enc(list(PROBES_PT))
    probes = pd.DataFrame([{"text": t_, "expected_sign": sg, "projection": float(v @ ax / np.linalg.norm(ax))}
                           for (t_, sg), v in zip(PROBES_PT.items(), pe)])
    probes.to_csv(HERE / "probe_words.csv", index=False)
    audit["probe_sign_correct"] = int((np.sign(probes.projection - out.projection.median()) == probes.expected_sign).sum())
    (HERE / "projection_audit.json").write_text(json.dumps(audit, indent=1, default=str))
    print(json.dumps(audit, indent=1, default=str))


if __name__ == "__main__":
    main()
