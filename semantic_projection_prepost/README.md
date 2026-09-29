# Pre vs post semantic projection (Q1 descriptors)

Pre/post version of `semantic_projection/` (which is untouched). Uses `data/consensus_pre_q1.csv` and `data/consensus_post_q1.csv`: the rater-consensus descriptors of Q1, split on `;` into one unit each (same cleaning as `pre_post/prepare_and_model.py`: 150 pre + 125 post descriptors, 49 participants with both interviews; participant 33 has no condition, 48 has no pre descriptors).

## Method
* `project_prepost.py`: Qwen3-Embedding-0.6B with the same emotional-valence instruction as the original; axis = mean of all (relief anchor - distress anchor) pairs; score = projection on the axis (positive = relief). **Axis = Portuguese translations of the anchor words** used in `semantic_projection.py` (distress: angustiado, perigo, apreensivo, angústia, tormento, agitado; relief: aliviado, confortado, segurança, acalmado, tranquilizado, à vontade; my translations, the descriptors being Portuguese). Descriptors are embedded as they are (multilingual model). Check: 10 Portuguese probe phrases all fall on the expected side of the median score (`probe_words.csv`; "ansiosa" alone is close to zero).
* `models_prepost.R [all|noreview]`: descriptor-level mixed model `projection ~ time * condition + (1 | participant)` with CR2 cluster-robust inference (as `word_level.R`), contrasts: change per condition, condition x time interaction (three pairs + planned VR vs Control), baseline and post condition differences; plus a participant-level paired analysis of the change in mean projection (t-tests, Wilcoxon, F-test and FDR-adjusted pairs for differences in change). `noreview` drops participants with rows flagged `needs_review`.
* `plot_projection_prepost.py`: figures.

Run order: `python3 project_prepost.py`, `Rscript models_prepost.R all` (and `noreview`), `python3 plot_projection_prepost.py`.

## Inference
The **main inferential model is the descriptor-level mixed model** (`projection ~ time * condition + (1 | participant)`, CR2 cluster-robust; BH-FDR within each family of contrasts (time effects per condition; the pooled change on its own; interaction contrasts; baseline/post differences; the planned contrast)). The participant-level paired analysis (t-tests, F-test on the change) is a **sensitivity analysis**, and so is the `noreview` subset.

## Results (all descriptors)
* Baseline (main model): the conditions do not differ before the intervention (all p > .2).
* Pre -> post (main model): relief increases overall (+0.101, p < .001; equal weights over conditions, its own family). By condition (FDR-adjusted over the three conditions): VR Only +0.142 (p = 0.007, FDR 0.022), VR Art +0.106 (p = 0.039, FDR 0.059), Control +0.054 (p = 0.158, FDR 0.158). Only VR Only is below .05 after FDR adjustment.
* Does the change differ by condition? (main model, interaction contrasts) VR Art vs Control +0.052 (p = .38), VR Only vs Control +0.088 (p = .14), VR Art vs VR Only -0.036 (p = .58); planned VR (Art + Only) vs Control +0.070 [-0.030, 0.169] (p = .16). Not significant.
* Sensitivity, participant-level paired analysis: overall +0.111 (p < .001); VR Art +0.119 (p = .026), VR Only +0.112 (p = .036), Control +0.102 (p = .068), FDR .054, .054 and .068 (VR Art, VR Only, Control); F-test for a difference in change p = .97. The equal weighting of participants makes the three conditions look alike; the mixed model weights descriptors, so participants with more descriptors count more.
* Sensitivity, `noreview` subset (31 paired participants): same picture, interaction never significant.

Outputs: `projection_descriptors.csv`, `projection_participant_time.csv`, `models_*_{subset}.csv`, `participant_change_*.csv`, `semantic_projection_prepost_scatter.*`, `semantic_projection_prepost_change.*`.
