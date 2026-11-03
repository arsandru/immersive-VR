# Handoff — immersive VR (2026-11-03)

Read this first, then `CLAUDE.md`. Update or replace this file at the end of each session.

## State

`main` is pushed at `2056cc3` (sentence-based PC labels, spaCy content-word filter for the word table, representative words on the figures). Untracked: `data/anxiety_fear_mechanism_items.csv` (not committed; commit only when asked).

## Repo reorganised (2026-11-03, not yet committed)

Main analyses stay at the top level (`pre_post/`, `semantic_projection_post/`, `q3/`, `all_questions/`, `topic_network_post_allquestions/`, plus `data/`, `transcribe/`). `semantic_projection_pre`, `semantic_projection_prepost`, `semantic_projection_fullresponse`, `topic_network_pre` and `topic_network_post` moved to `supplementary/` with `git mv`. Imports of `q3` and `all_questions` are unchanged (repo root on `sys.path`); path constants and docs of the moved folders were updated. Checked: all scripts import, all path constants resolve, `pre_post/check_topic_consistency.py` passes (ALL CONSISTENT), `pc4_presence.py` reproduces its numbers. Not re-run: the heavy analyses themselves (BERTopic fits, permutation tests) and the R scripts; stale absolute paths remain inside some tracked `outputs/run_summary.json` files until those are regenerated.

## PC4 vs presence: done (not yet committed)

The presence file was corrected; `participant` (P0001, ...) now maps to the interview participant number (condition labels agree for all 58). `topic_network_post_allquestions/pc4_presence.py` ran: 51 participants (17 per group); primary (user decision): pooled Spearman rho 0.31 [0.04, 0.54], p = 0.030, one test so no correction; confound checks: within group (group means removed) rho -0.09 [-0.37, 0.20], p = 0.53; per group -0.08 / -0.07 / -0.15. Presence has a floor (9 of 17 Control zeros, none in VR). Reading: the pooled link is the group difference. Results are in the folder README. No further analyses were requested (PC6, other columns).

## Offered, not requested

- Merge `sinto` into `sentir` in the word table (one-line change).
- Port the topic-free PC scripts and figures to `supplementary/topic_network_pre/` and `supplementary/topic_network_post/` (still topic-based).
- The user's earlier question on pre-registration (which `semantic_projection_*` folders and contrasts are pre-registered; whether combined VR vs Control is planned) is unanswered.

## Rules to remember

See `CLAUDE.md`: Portuguese only; Qwen3-Embedding-0.6B only (no 4B or bigger on the 16 GB MacBook); no restart of killed runs without asking; commit and push only when asked.
