# Immersive VR: working rules

Analysis workspace for a paper. Portuguese-language interviews from a three-group study (pre and post). The README is stale; trust this file and the per-folder READMEs, and check the code when they disagree.

## Design
- Groups (`condition_number` in `data/participant_conditions.csv`, joined on `enrollment_number`): 1 = VR Art, 2 = VR Only, 3 = Control (the reference level).
- Inference is group based: VR Art, VR Only and Control. Planned contrasts are VR Art vs Control, VR Only vs Control and VR Art vs VR Only, plus the combined VR vs Control contrast where a script already defines it. Do not invent new groupings or contrasts.
- Unit of analysis is the participant. Words, descriptors and sentences are nested within participants, so use participant-clustered inference (CR2, mixed models) or permutation with participant as the unit.
- Known data quirks: participant 33 has no condition (excluded); participant 48 has no pre descriptors; 49 participants have both timepoints. Do not change exclusions silently.
- All data used lives in `data/`. Never read data from anywhere else.

## Confirmatory vs exploratory
- **Only the semantic projection is pre-registered** (`semantic_projection_pre/`, `_post/`, `_prepost/`; which of these and which contrasts are registered is to be confirmed with me). Its specification is fixed: Qwen3-Embedding-0.6B with the stated instruction, the Portuguese distress and relief anchors, and the models in `models*.R`. Do not change it, and do not add analyses to it, without asking first.
- Everything else is exploratory: topic analyses (`q3`, `all_questions`, `topic_network_*`), `pre_post`. Label such results "exploratory" in reports, figure captions and any text intended for the paper.
- A new test is exploratory unless you are told otherwise. Say which family it belongs to and what the multiplicity correction covers (BH-FDR within the family, as the existing scripts do).

## Statistical discipline
- Between-group comparisons are the target of the study and are always allowed: test them directly (contrasts, group × time) and report them. The only thing to avoid is inferring a group difference from one group's test being significant and another's not ("significant in VR, not in Control"); that is not a between-group test.
- Report estimates with intervals and the correction used, not only p-values. Say "no evidence of a difference" for a null, not "no effect".
- Small samples (about 15 to 20 per group; 16 to 23 transitions per flow graph): call such results descriptive and do not over-interpret.
- Report the n actually analysed for every result (participants and units).
- Do not drop rows or participants, tune thresholds, or change a clustering after seeing results. If a sensitivity analysis is wanted, add it as a labelled sensitivity run with its own tag.

## Language
- Text stays in Portuguese for every analysis: no translation, no English paraphrase fed to models. Stop words, stems and lexicons must be Portuguese.
- Topic labels and topic interpretations are human-written (`timepoint_topic_labels.py`, `prepost_common.py` and similar). Do not rewrite or regenerate labels without asking, and never let a script overwrite them.

## Reproducibility
- Run from the repository root. Use a project `.venv` (not yet present on this machine, so create one from the folder `requirements.txt` files and ask before installing anything).
- Seeds: BERTopic/UMAP use `random_state=42`; permutation scripts take `--seed`. Do not remove seeds. When you add stochastic code, set a seed and record it.
- `pre_post/timepoint_topic_labels.py` is the single source of truth for BERTopic parameters and labels. After changing anything that touches topics, run `python3 pre_post/check_topic_consistency.py`.
- Generated CSV, PDF and PNG outputs are tracked in git. Re-running a script overwrites the committed results, so check `git status` and `git diff --stat` afterwards and tell me what changed numerically. Do not regenerate outputs speculatively.
- Never overwrite `data/`. Treat the files there as read-only.

## Participant privacy
Interviews are personal data. Do not print long verbatim passages or participant identifiers into chat, commit messages or new files unless needed. Do not copy data outside the repo, and do not send it to external services.

## Working rules
- Commit and push only when I ask. Check `git status` first, stage specific paths (not `git add -A`), and do not commit `.venv`, `__pycache__` or `data/audio/`.
- Before large re-runs (BERTopic refits, 1000-refit null distributions, which take minutes), say what will run and what it will overwrite.
- Keep the folder READMEs in step with any result you change; flag numbers in the text that no longer match the outputs.
- When a result is surprising, check the data and the code path before explaining it.
- Say plainly when something was not verified, skipped or failed.
