# Handoff — immersive VR (2026-11-03, end of session)

Read this first, then `CLAUDE.md`. Update or replace this file at the end of each session.

## State

`main` is pushed at `78757cc` (repo reorganisation, consistent state). `36a39fb` just before it is a partial commit (renames only, paths and docs not yet updated), so do not check out or bisect to it. Untracked: `data/anxiety_fear_mechanism_items.csv` (the corrected presence file; check `git status` before committing anything, and commit it only when asked). This `HANDOFF.md` edit is not yet committed.

## Repo layout (reorganised this session)

Top level, main analyses: `pre_post/`, `semantic_projection_post/`, `q3/`, `all_questions/`, `topic_network_post_allquestions/`, plus `data/` and `transcribe/`. In `supplementary/`: `semantic_projection_pre`, `semantic_projection_prepost`, `semantic_projection_fullresponse`, `topic_network_pre`, `topic_network_post` (207 tracked files, moved with `git mv`). Scripts import shared helpers as `q3...` and `all_questions...` with the repo root on `sys.path`; run everything from the repo root. Do not move folders or add top-level analysis folders without asking (in `CLAUDE.md`).

Checked after the move: all scripts import, path constants resolve, `python3 pre_post/check_topic_consistency.py` ends with ALL CONSISTENT, `pc4_presence.py` reproduces its numbers. Not re-run: the heavy analyses (BERTopic fits, permutation tests) and the R scripts. Some tracked `outputs/run_summary.json` files still hold stale absolute paths until those analyses are regenerated. README.md (top level) now has a main table and a supplementary table.

## PC4 vs presence (committed, exploratory)

Script `topic_network_post_allquestions/pc4_presence.py`; outputs `outputs/pc4_presence.csv/.pdf/.png`; results table in that folder's README. The `participant` column (P0001, ...) maps to the interview participant number (condition labels agree for all 58). 51 participants (17 per group). Participant mean PC4 score vs presence (0-10), Spearman:

- **Primary (your decision), pooled:** rho 0.31 [0.04, 0.54], permutation p = 0.030. One test, so no multiplicity correction.
- **Confound checks** (sensitivity analyses, reported whatever they show): within group with group means removed rho -0.09 [-0.37, 0.20], p = 0.53; per group -0.08 (VR Art), -0.07 (VR Only), -0.15 (Control), descriptive.
- Presence has a floor: 9 of 17 Control participants score 0, none in the VR groups.
- Reading: the pooled link is carried by the group difference, with no evidence of a relationship within groups. Not a statement about mechanism.

## Open items (none requested yet)

- Merge `sinto` into `sentir` in the word table (one-line change).
- Port the topic-free PC scripts and figures to `supplementary/topic_network_pre/` and `supplementary/topic_network_post/` (still topic-based).
- Pre-registration question is unanswered: which `semantic_projection_*` folders and contrasts are registered, and whether combined VR vs Control is planned. `CLAUDE.md` says this is still to be confirmed.
- Regenerate the heavy analyses to refresh stale paths in `outputs/run_summary.json` (ask before large re-runs; they overwrite tracked outputs).

## Rules to remember

See `CLAUDE.md`: Portuguese only; Qwen3-Embedding-0.6B only (no 4B or bigger on the 16 GB MacBook); no restart of killed runs without asking; commit and push only when asked, staging specific paths (not `git add -A`; that command failed once this session on stale folder names and left a partial commit); label new tests exploratory; `data/` is read-only.

## Other repo: `/Users/razvan/wl_25`

See that repo's `HANDOFF.md` (manuscript alignment via tika suggestions; many marks pending acceptance).
