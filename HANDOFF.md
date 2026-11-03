# Handoff — immersive VR (2026-11-03)

Read this first, then `CLAUDE.md`. Update or replace this file at the end of each session.

## State

`main` is pushed at `2056cc3` (sentence-based PC labels, spaCy content-word filter for the word table, representative words on the figures). Untracked: `data/anxiety_fear_mechanism_items.csv` (not committed; commit only when asked).

## Blocked task: PC4 vs presence

Request: does each participant's mean PC4 score (full-interview sentence embedding space, `topic_network_post_allquestions/`) correlate with their presence score in `data/anxiety_fear_mechanism_items.csv`?

**Blocker.** The CSV identifies participants as S01–S58, 58 rows (Control 20: S01–S20; VR Art 19: S21–S39; VR Only 19: S40–S58). The interview data use participant ids 1–60 (`enrollment_number` in `data/participant_conditions.csv`). The user said "S01–S58 corresponds to the participant id from the interview data", but joining on that id gives condition agreement in only 21 of 56 matched ids (about chance), and the S-numbers are grouped by condition, so they cannot be enrollment ids. The user then agreed it does not work and is sorting out a key. Do not run the analysis until a key (S-number to interview participant id) exists and the condition labels agree for every matched participant.

**Analysis as agreed once the key exists (one script, no extras).**
1. Predictor: participant mean PC4 score from the existing PCA (sentence level, participant mean as in the PC4 group test; 51 usable participants minus any without presence).
2. Spearman correlation of mean PC4 with presence; rho, participant-bootstrap CI, permutation p. One pre-specified test, no multiplicity correction. Exploratory.
3. Group confound: report the pooled correlation and a within-group correlation (rank correlation of residuals after removing group means), plus per group. Check how many participants score 0 on presence (floor effect, mostly Control).
4. Do not add PC6, other PCs or other columns.

## Offered, not requested

- Merge `sinto` into `sentir` in the word table (one-line change).
- Port the topic-free PC scripts and figures to `topic_network_pre/` and `topic_network_post/` (still topic-based).
- The user's earlier question on pre-registration (which `semantic_projection_*` folders and contrasts are pre-registered; whether combined VR vs Control is planned) is unanswered.

## Rules to remember

See `CLAUDE.md`: Portuguese only; Qwen3-Embedding-0.6B only (no 4B or bigger on the 16 GB MacBook); no restart of killed runs without asking; commit and push only when asked.
