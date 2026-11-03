# Full-response semantic projection (post interview)

**Exploratory** (not pre-registered). One score per participant from the whole post interview: the three answers of `data/immerse_interviewPost.csv` (Q1 "Como se sente agora?", Q2 difference from before, Q3 aspects of care that mattered) concatenated (non-empty answers only, question order, space-joined), embedded as one text.

## Method
* `project.py`: Qwen3-Embedding-0.6B, the same emotional-valence instruction and Portuguese distress-relief anchors as `../semantic_projection_post/` (positive = relief). The instruction says "word or short phrase", so it is a loose fit for texts of median 56 words (range 7 to 274). Writes `projection_participants.csv`, `probe_words.csv` (9 of 10 probes have the expected sign), `projection_audit.json`.
* Participants are keyed by `enrollment_number`. Excluded: 33 and 11 (no condition) and all participants with three empty answers (6, 8, 10, 11, 23, 30, 48, 55). Left: 51 participants (17 per group); 28 answered all three questions, 22 two, 1 one. Q2 is empty for 29 of 60 (median 1 word), so the concatenated text is mostly Q1 and Q3.
* `models.R`: `lm(projection ~ condition)` with weighted effect coding (weights = n participants), omnibus F, pairwise contrasts (emmeans, BH-FDR over 3 pairs), Cohen's d, diagnostics. Full output in `log.txt`, files `models_*.csv`, `diagnostics.pdf`, `semantic_projection_fullresponse.pdf/svg`.

Run: `python3 project.py`, `Rscript models.R`.

## Results (51 participants)
* No condition difference: omnibus F p = 0.46. Group means 0.166 / 0.197 / 0.228 (Control / VR Art / VR Only). Pairwise FDR p = 0.54 for all three contrasts (VR Art vs Control +0.030, VR Only vs Control +0.062, d = 0.21 and 0.41).
* Residuals depart from normality (Shapiro-Wilk p = 0.003); variances are similar (Bartlett p = 0.80).
* The projection mixes all three questions, including Q3 (care received, not emotional state).
