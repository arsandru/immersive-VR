#!/usr/bin/env python3
"""Full-response semantic projection (post interview): one text per participant.

The three answers of data/immerse_interviewPost.csv (Q1 how do you feel now, Q2 difference from before, Q3 aspects of care
that mattered) are concatenated (non-empty answers only, in question order, joined by a space), embedded as ONE text with
Qwen3-Embedding-0.6B and the emotional-valence instruction used in
semantic_projection_post, and projected on the same Portuguese distress-relief axis (positive = relief). Portuguese throughout.

Participants are keyed by enrollment_number (data/participant_conditions.csv); participants without a condition or with
three empty answers are excluded and listed in projection_audit.json.
Outputs: projection_participants.csv, probe_words.csv, projection_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_INSTRUCTION = "Instruct: Represent the emotional valence and intensity of this word or short phrase.\nText: "
DISTRESS_PT = ["angustiado", "perigo", "apreensivo", "angústia", "tormento", "agitado"]
RELIEF_PT = ["aliviado", "confortado", "segurança", "acalmado", "tranquilizado", "à vontade"]
PROBES_PT = {"ansiosa": -1, "com medo": -1, "preocupada": -1, "muito nervosa": -1, "a ansiedade aumenta": -1,
             "calma": 1, "relaxada": 1, "tranquila": 1, "aliviada": 1, "sinto-me segura": 1}
COND_NAMES = {1: "VR Art", 2: "VR Only", 3: "Control"}

model = SentenceTransformer(MODEL_ID)


def enc(texts):
    return model.encode([f"{EMBED_INSTRUCTION}{t}" for t in texts], convert_to_numpy=True, show_progress_bar=False)


def main() -> None:
    d = pd.read_csv(DATA / "immerse_interviewPost.csv")
    cols = list(d.columns[1:])
    cond = pd.read_csv(DATA / "participant_conditions.csv").set_index("enrollment_number")["condition_number"].to_dict()
    audit: dict = {"model": MODEL_ID, "questions": [c.strip() for c in cols], "participants_in_file": int(len(d))}
    txt = d[cols].fillna("").astype(str).apply(lambda r: " ".join(x.strip() for x in r if x.strip()), axis=1)
    audit["empty_answers_per_question"] = {c.strip(): int((d[c].fillna("").astype(str).str.strip() == "").sum()) for c in cols}
    out = pd.DataFrame({"participant_id": d.Participant, "text": txt})
    out["n_answers"] = d[cols].fillna("").astype(str).apply(lambda r: sum(bool(x.strip()) for x in r), axis=1)
    out["n_words"] = out.text.str.split().str.len()
    audit["no_condition"] = sorted(out.participant_id[~out.participant_id.isin(cond)].tolist())
    audit["all_empty"] = sorted(out.participant_id[out.n_answers == 0].tolist())
    out = out[out.participant_id.isin(cond) & (out.n_answers > 0)].reset_index(drop=True)
    out["condition_label"] = out.participant_id.map(lambda p: COND_NAMES[int(cond[p])])
    audit["participants"] = int(len(out))
    audit["by_condition"] = out.condition_label.value_counts().to_dict()
    audit["n_answers_distribution"] = out.n_answers.value_counts().sort_index().to_dict()
    p, n = enc(RELIEF_PT), enc(DISTRESS_PT)
    ax = np.array([a - b for a in p for b in n]).mean(axis=0)
    unit = ax / np.linalg.norm(ax)
    out["projection"] = enc(out.text.tolist()) @ unit
    out.to_csv(HERE / "projection_participants.csv", index=False)
    pe = enc(list(PROBES_PT))
    probes = pd.DataFrame([{"text": t, "expected_sign": s, "projection": float(v @ unit)} for (t, s), v in zip(PROBES_PT.items(), pe)])
    probes.to_csv(HERE / "probe_words.csv", index=False)
    audit["probe_sign_correct"] = int((np.sign(probes.projection - out.projection.median()) == probes.expected_sign).sum())
    (HERE / "projection_audit.json").write_text(json.dumps(audit, indent=1, default=str, ensure_ascii=False))
    print(json.dumps(audit, indent=1, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
