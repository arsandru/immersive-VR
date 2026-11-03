# Semantic projection at the pre-interview timepoint

The condition comparison of `semantic_projection/` (word-level mixed model with CR2 cluster-robust inference, participant-level sensitivity), run on the **pre-interview** Q1 descriptors only, on all participants observed at this timepoint. Self-contained: the scripts detect the timepoint from the folder name (`semantic_projection_pre` / `semantic_projection_post`); the only external dependency is the cleaning step `pre_post/prepare_and_model.py` (splits `data/consensus_pre_q1.csv` into one descriptor per row).

## Method
* `project.py`: Qwen3-Embedding-0.6B with the original emotional-valence instruction. Axis = mean of all (relief - distress) anchor pairs, score = projection on the axis (relief positive). Axis = Portuguese translations of the original anchors (angustiado, perigo, apreensivo, angústia, tormento, agitado / aliviado, confortado, segurança, acalmado, tranquilizado, à vontade; my translations). The figure's y coordinate is PC1 of the residual variance of this timepoint's descriptors. Writes `projection_descriptors.csv`, `probe_words.csv`, `projection_audit.json`.
* `models.R [all|noreview]`: **main inferential model** = descriptor-level mixed model `projection ~ condition + (1 | participant)`, CR2, pairwise Wald tests (BH-FDR over three pairs), model means; **sensitivity** = participant-mean analysis (F test, FDR-adjusted pairs, Cohen's d). `noreview` drops participants with rows flagged `needs_review`.
* **Headline figure `semantic_projection_pre.pdf/.svg`** (written by `models.R`; the `noreview` sensitivity run writes `semantic_projection_pre_noreview.pdf`): same design as `semantic_projection/semantic_projection_primary_final.pdf`, model-based without the scatter: model-estimated condition means from the descriptor-level mixed model, standardised by the residual SD of the model, with 95% CIs, dashed line at 0, brackets only for pairs with BH-FDR-adjusted p < .05 (none here).
* `plot.py`: `semantic_projection_pre_scatter.pdf/.svg/.png`, the descriptor scatter in the layout of the original scatter figure (panels Control, VR Art, VR Only; coloured line and band = model mean with 95% CI; grey band = median absolute score). Supplementary.

Run: `python3 project.py`, `Rscript models.R all` (and `noreview`; needs ggplot2), `python3 plot.py`.

## Results ( 51 participants, 150 descriptors)
* No condition differs before VR: main model (descriptor-level mixed model) p = 1.00, 0.33, 0.22 (VR Art v Control, VR Only v Control, VR Art v VR Only); sensitivity, participant-level F-test p = 0.97; participant means 0.033 / 0.038 / 0.024 (Control, VR Art, VR Only).
* Same result with `needs_review` participants dropped (sensitivity).
