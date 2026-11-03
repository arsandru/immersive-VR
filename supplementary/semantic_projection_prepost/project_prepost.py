#!/usr/bin/env python3
"""Pre vs post semantic projection of the Q1 descriptors on the distress-vs-relief axis.

Same method as semantic_projection/semantic_projection.py: Qwen3-Embedding-0.6B with the same
emotional-valence instruction, axis = mean over all (relief anchor - distress anchor) pairs, score
= projection of the embedded text on that axis (positive = relief, negative = distress).

Differences: the units are the Portuguese rater-consensus descriptors of Q1 (data/consensus_pre_q1.csv,
consensus_post_q1.csv; same cleaning as pre_post/prepare_and_model.py), embedded as they are
(the model is multilingual).  The axis uses Portuguese translations of the anchor words of the existing analysis
(the descriptors are Portuguese).

Outputs (this folder):
  projection_descriptors.csv   one row per descriptor, with participant, condition, time
  projection_participant_time.csv  participant x time means
  probe_words.csv / projection_audit.json   validity checks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pre_post"))
from prepare_and_model import load_units  # noqa: E402

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_INSTRUCTION = "Instruct: Represent the emotional valence and intensity of this word or short phrase.\nText: "

DISTRESS_PT = ["angustiado", "perigo", "apreensivo", "angústia", "tormento", "agitado"]
RELIEF_PT = ["aliviado", "confortado", "segurança", "acalmado", "tranquilizado", "à vontade"]
PROBES_PT = {  # word/phrase: expected sign (+ relief, - distress)
    "ansiosa": -1, "com medo": -1, "preocupada": -1, "muito nervosa": -1, "a ansiedade aumenta": -1,
    "calma": 1, "relaxada": 1, "tranquila": 1, "aliviada": 1, "sinto-me segura": 1,
}

model = SentenceTransformer(MODEL_ID)


def enc(texts):
    return model.encode([f"{EMBED_INSTRUCTION}{t}" for t in texts], convert_to_numpy=True, show_progress_bar=False)


def make_axis(pos, neg):
    p, n = enc(pos), enc(neg)
    return np.array([a - b for a in p for b in n]).mean(axis=0)  # relief minus distress: + = relief


def main() -> None:
    audit: dict = {}
    u = load_units(audit)
    E = enc(u["sentence"].tolist())
    out = u[["document_id", "participant_id", "condition_label", "time", "sentence_number", "sentence",
             "needs_review"]].copy()
    ax = make_axis(RELIEF_PT, DISTRESS_PT)
    out["projection"] = E @ ax / np.linalg.norm(ax)

    # 2-D coordinates for the plot: x = projection, y = PC1 of the residuals (as in the original figure)
    unit = ax / np.linalg.norm(ax)
    resid = E - np.outer(E @ unit, unit)
    from sklearn.decomposition import PCA
    out["y_pc1_residual"] = PCA(n_components=1, random_state=0).fit_transform(resid).ravel()
    out.to_csv(HERE / "projection_descriptors.csv", index=False)

    pt = (out.groupby(["participant_id", "condition_label", "time"])
          .agg(mean_projection=("projection", "mean"), n_descriptors=("sentence", "size"),
               any_needs_review=("needs_review", "any")).reset_index())
    pt.to_csv(HERE / "projection_participant_time.csv", index=False)

    # validity: probe phrases (expected sign relative to the median descriptor score)
    pe = enc(list(PROBES_PT))
    probes = pd.DataFrame([{"text": t, "expected_sign": s, "projection": float(v @ ax / np.linalg.norm(ax))}
                           for (t, s), v in zip(PROBES_PT.items(), pe)])
    probes.to_csv(HERE / "probe_words.csv", index=False)
    audit["probe_sign_correct"] = int((np.sign(probes.projection - out.projection.median()) == probes.expected_sign).sum())
    (HERE / "projection_audit.json").write_text(json.dumps(audit, indent=1, default=str))
    pd.set_option("display.width", 200)
    print(probes.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in audit.items() if "probe" in k}, indent=1))
    print(out.groupby(["time", "condition_label"]).projection.agg(["count", "mean", "std"]).round(3))


if __name__ == "__main__":
    main()
