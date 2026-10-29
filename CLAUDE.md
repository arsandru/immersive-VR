# Immersive VR: working rules

Analysis workspace for a paper. Portuguese-language interviews from a three-group study (pre and post). The README is stale; trust this file and the per-folder READMEs, and check the code when they disagree.

## Design
- Groups (`condition_number` in `data/participant_conditions.csv`, joined on `enrollment_number`): 1 = VR Art, 2 = VR Only, 3 = Control (the reference level).
- Inference is group based: VR Art, VR Only and Control. Planned contrasts are VR Art vs Control, VR Only vs Control and VR Art vs VR Only, plus the combined VR vs Control contrast where a script already defines it. Do not invent new groupings or contrasts.
- Unit of analysis is the participant. Words, descriptors and sentences are nested within participants, so use participant-clustered inference (CR2, mixed models) or permutation with participant as the unit.
- What the conditions are: **Control** waited in a pre-surgery room with no content available; **VR Only** watched nature videos in VR; **VR Art** experienced an immersive art installation. Only the VR groups can describe a stimulus, so a content difference between VR and Control is partly expected. Treat "the VR groups talk about the video/nature/art" as a possible trivial explanation and check it (a stimulus-word sensitivity run) before reading a difference as a change in feeling.
- Known data quirks: participant 33 has no condition (excluded); participant 48 has no pre descriptors; 49 participants have both timepoints. In `data/immerse_interviewPost.csv` (60 rows) participants 11 and 33 have no condition and 8 participants have all three answers empty (6, 8, 10, 11, 23, 30, 48, 55); 51 remain (17 per group). Q2 is empty for 29 of 60 and its median length is 1 word. Do not change exclusions silently.
- All data used lives in `data/`. Never read data from anywhere else.

## Confirmatory vs exploratory
- **Only the semantic projection is pre-registered** (`semantic_projection_pre/`, `_post/`, `_prepost/`; which of these and which contrasts are registered is still to be confirmed with me). The analysis is fixed: the stated instruction, the Portuguese distress and relief anchors, and the models in `models*.R`. Do not add analyses to it without asking. **The embedding model itself is not pre-registered and may be changed**, but only for validity reasons (see "Models and hardware"), never because of what it does to the group contrasts.
- Everything else is exploratory: topic analyses (`q3`, `all_questions`, `topic_network_*`), `pre_post`. Label such results "exploratory" in reports, figure captions and any text intended for the paper.
- A new test is exploratory unless you are told otherwise. Say which family it belongs to and what the multiplicity correction covers (BH-FDR within the family, as the existing scripts do).

## Statistical discipline
- Between-group comparisons are the target of the study and are always allowed: test them directly (contrasts, group × time) and report them. The only thing to avoid is inferring a group difference from one group's test being significant and another's not ("significant in VR, not in Control"); that is not a between-group test.
- Report estimates with intervals and the correction used, not only p-values. Say "no evidence of a difference" for a null, not "no effect".
- Small samples (about 15 to 20 per group; 16 to 23 transitions per flow graph): call such results descriptive and do not over-interpret.
- Report the n actually analysed for every result (participants and units).
- Do not drop rows or participants, tune thresholds, or change a clustering after seeing results. If a sensitivity analysis is wanted, add it as a labelled sensitivity run with its own tag. Say when a rule (a word list, a threshold) was written after looking at the data; such analyses are exploratory.
- Condition is coded with **weighted effect coding (WEC)** in the semantic projection models (intercept = sample-weighted grand mean, coefficients = group deviations from it, reference Control coded -n_k / n_Control). It is equivalent to treatment coding for contrasts (the script checks fitted values). Compare models across codings with fitted values or variance components, not REML log-likelihoods (those are not comparable across codings).
- Pairwise contrasts for the descriptor-level mixed models use CR2 cluster-robust variance (`clubSandwich` `Wald_test`, HTZ) because of the 16 to 17 clusters per group; `emmeans` is used for the participant-level models and for estimated means. Do not swap the method without saying so.
- Every model script must write its full console output (model summary, variance components, ICC, residual diagnostics, `sessionInfo()`) to a log that the script itself writes, so logs cannot go stale.
- Choosing a model, anchor set or filter by looking at which one gives the clearest group result is not allowed. Choose on measurement validity (e.g. negation handling, known-valence phrases) and decide in advance how the alternatives will be reported (as sensitivity runs).

## Language
- Text stays in Portuguese for every analysis: no translation, no English paraphrase fed to models. Stop words, stems and lexicons must be Portuguese.
- Topic labels and topic interpretations are human-written (`timepoint_topic_labels.py`, `prepost_common.py` and similar). Do not rewrite or regenerate labels without asking, and never let a script overwrite them.
- Word-level tables use lower-cased Portuguese tokens with function words removed; keep evaluative words such as *bem*, *bom*, *boa* (the stop-word list in `q3/` also contains them, so slice it before using it). No Portuguese lemmatizer is installed; ask before installing one.

## Reproducibility
- Run from the repository root. Use a project `.venv` (not yet present on this machine, so create one from the folder `requirements.txt` files and ask before installing anything).
- Seeds: BERTopic/UMAP use `random_state=42`; permutation scripts take `--seed`. Do not remove seeds. When you add stochastic code, set a seed and record it.
- Numpy on this machine raises spurious overflow warnings with `E @ E.T`; use `np.einsum("id,jd->ij", E, E)` for similarity matrices (as `embedding_space_tests.py` does). Avoid a DataFrame column called `map` (it shadows the pandas method).
- If a rerun changes only the row order of a CSV, restore the file with `git checkout` instead of committing the noise.
- `pre_post/timepoint_topic_labels.py` is the single source of truth for BERTopic parameters and labels. After changing anything that touches topics, run `python3 pre_post/check_topic_consistency.py`.
- Generated CSV, PDF and PNG outputs are tracked in git. Re-running a script overwrites the committed results, so check `git status` and `git diff --stat` afterwards and tell me what changed numerically. Do not regenerate outputs speculatively.
- Never overwrite `data/`. Treat the files there as read-only.

## Models and hardware
- The machine is a 16 GB MacBook Air (Apple M5). `SentenceTransformer` uses the GPU (MPS) automatically.
- **Current embedding model everywhere: Qwen3-Embedding-0.6B** with the emotional-valence instruction. Qwen3-Embedding-4B was audited (better negation handling, validation accuracy 1.00 vs 0.92; Spearman 0.92 with 0.6B) and tried for the post projection, then reverted to 0.6B: it needs about 16 GB in fp32, was killed for low memory, and bf16 and fp16 are unusable on this machine (bf16 degraded the scores, rho 0.65 with fp32; fp16 gave NaNs). Do not run 4B or larger models here unless I ask; if memory is short, say so instead of retrying. Background runs are killed under memory pressure and must not be restarted without asking.
- Long texts (full responses, up to 274 words) need small batches on MPS (`batch_size=2`) or they run out of GPU memory.
- Model audit facts (0.6B, Portuguese): known-valence phrases are classified well, leave-one-anchor-out axes correlate >= 0.996, masculine vs feminine forms differ by about 0.02, negation is the weak spot (all six test pairs moved the expected way, but "não estou ansiosa" barely moved, +0.05 vs +0.39 for 4B), and individual descriptor scores differ between models (rho 0.82 with multilingual-e5-large-instruct). The anchors are translations by me and have not been checked by a native speaker; there is no human valence rating as a criterion.
- Negation-filtering the descriptors was tried and rejected (removes real signal, adds a researcher choice); do not reintroduce it unless asked.

## Embedding-space analyses and reading the PCs
- Headline tests (PERMANOVA on cosine distances, dispersion, the nearest-neighbour mixing test) use the full 1024-dimension embedding with no dimension reduction. PCA is used only to describe where differences lie (first 10 PCs, permutation ANOVA on participant scores, BH over 10). t-SNE was tried and removed: its axes and distances are not interpretable, it is unstable across seeds and perplexities, and it showed no visible condition separation. Kernel PCA has not been run; if it is, fix the bandwidth in advance (median heuristic).
- The PC4 x PC6 map picks axes that separate the conditions, so its visual separation is partly by construction; say so when showing it.
- **Read PCs from sentences, not from topics.** Use representative sentences (one per participant) and the word-association table (`pc_sentence_interpretation.py`), plus nuisance checks (sentence length, question, within-question tests) and a stimulus-word sensitivity check (`pc_stimulus_sensitivity.py`). BERTopic topics only decide which sentences are kept (5 outlier sentences are dropped) and label the map; do not build interpretation by superimposing topics.
- Current reading (exploratory): **PC4 is the finding** (FDR 0.003; VR Art high, Control low, VR Only between): evaluating an experience ("Experience & senses") vs reporting one's state about the procedure and care ("Procedure & care"). It holds within Q1, survives removing stimulus-word sentences (VR Art vs Control and vs VR Only; VR Only vs Control does not), and describes content, not an emotion rating. **PC6 is tentative**: mixed ends, disappears when stimulus words are removed. A nearest-neighbour test agrees (VR Art vs Control only, +5 points above chance, Holm 0.022) but uses the same embedding, so it is not independent evidence.
- Re-labelling a PC: update `PC_END_LABELS` in `pc_end_evidence.py`, rerun it and `plot_semantic_space.py`, and update the README.

## Participant privacy
Interviews are personal data. Do not print long verbatim passages or participant identifiers into chat, commit messages or new files unless needed. Do not copy data outside the repo, and do not send it to external services.

## Working rules
- Remote: `https://github.com/arsandru/immersive-VR` (branch `main`). Commit and push only when I ask. Check `git status` first, stage specific paths (not `git add -A`), and do not commit `.venv`, `__pycache__` or `data/audio/`.
- Before large re-runs (BERTopic refits, 1000-refit null distributions, which take minutes), say what will run and what it will overwrite.
- Keep the folder READMEs in step with any result you change; flag numbers in the text that no longer match the outputs.
- When a result is surprising, check the data and the code path before explaining it.
- Say plainly when something was not verified, skipped or failed.
