# Semantic projection at the post-interview timepoint

The condition comparison of `semantic_projection/` (word-level mixed model with CR2 cluster-robust inference, participant-level sensitivity), run on the **post-interview** Q1 descriptors only, on all participants observed at this timepoint. Self-contained: the scripts detect the timepoint from the folder name (`semantic_projection_pre` / `semantic_projection_post`); the only external dependency is the cleaning step `pre_post/prepare_and_model.py` (splits `data/consensus_post_q1.csv` into one descriptor per row).

## Method
* `project.py`: Qwen3-Embedding-0.6B with the original emotional-valence instruction. Axis = mean of all (relief - distress) anchor pairs, score = projection on the axis (relief positive). Axis = Portuguese translations of the original anchors (angustiado, perigo, apreensivo, angústia, tormento, agitado / aliviado, confortado, segurança, acalmado, tranquilizado, à vontade; my translations). The figure's y coordinate is PC1 of the residual variance of this timepoint's descriptors. Writes `projection_descriptors.csv`, `probe_words.csv`, `projection_audit.json`.
* `models.R [all|noreview]`: **main inferential model** = descriptor-level mixed model `projection ~ condition + (1 | participant)`, CR2, pairwise Wald tests (BH-FDR over three pairs), model means; **sensitivity** = participant-mean analysis (F test, FDR-adjusted pairs, Cohen's d). `noreview` drops participants with rows flagged `needs_review`.
* **Headline figure `semantic_projection_post.pdf/.svg`** (written by `models.R`; the `noreview` sensitivity run writes `semantic_projection_post_noreview.pdf`): same design as `semantic_projection/semantic_projection_primary_final.pdf`, model-based without the scatter: model-estimated condition means from the descriptor-level mixed model, standardised by the residual SD of the model, with 95% CIs, dashed line at 0, brackets only for pairs with BH-FDR-adjusted p < .05 (none here).
* `plot.py`: `semantic_projection_post_scatter.pdf/.svg/.png`, the descriptor scatter in the layout of the original scatter figure (panels Control, VR Art, VR Only; coloured line and band = model mean with 95% CI; grey band = median absolute score). Supplementary.

Run: `python3 project.py`, `Rscript models.R all` (and `noreview`; needs ggplot2), `python3 plot.py`.

## Results ( 50 participants, 125 descriptors)
* No condition differs after VR: main model (descriptor-level mixed model) p = 0.34, 0.52, 0.87 (VR Art v Control, VR Only v Control, VR Art v VR Only); sensitivity, participant-level F-test p = 0.79; participant means 0.119 / 0.157 / 0.141 (Control, VR Art, VR Only). VR Art vs Control at participant level d = 0.27 (not significant).
* Same result with `needs_review` participants dropped (sensitivity).
