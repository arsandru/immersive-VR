# Pre vs post VR: presentation report

**Data.** Rater-consensus descriptors of Q1 ("how do you feel now?") from the pre and post interviews: 150 pre and 125 post descriptors, 49 participants with both interviews (VR Art 17, VR Only 17, Control 15). Portuguese text, embedded with Qwen3-Embedding-0.6B. Conditions do not differ before the intervention (baseline: topic composition p = .52, semantic space p = .65, flow graphs p = .37). Multiple comparisons: Benjamini-Hochberg FDR throughout.

**Sources.** Slides 1-3 use the Q1 descriptors at both timepoints (`pre_post/`, `semantic_projection_prepost/`); slide 4 uses the pre descriptors (`topic_network_pre/`); slide 5 uses the full post-interview answers (Q1-Q3 sentences, `topic_network_post/`).

**One-sentence message.** From pre to post, everyone's language moves from worry and anticipation toward calm; VR Art and VR Only shift significantly, Control does not reach significance, but the size of the change does not reliably differ between conditions.

---

## Slide 1 - Core networks: what participants' talk revolves around (`core_network_pre_vs_post.pdf`)

**What it shows.** One network per condition (columns) before (top row) and after (bottom row). Nodes are the 11 topics found in the descriptors; arrows are observed transitions between consecutive descriptors in one participant's answer (thicker = more often). Node size and colour are PageRank, the share of "attention" a topic collects when following the transitions; the same scale is used in all six panels. The most central topic sits in the middle, the next three on an inner ring, the rest outside.

**How.** One BERTopic model on pre and post descriptors together, so topics mean the same at both times. PageRank (damping 0.85) is computed per condition and timepoint from that condition's transition counts.

**What it shows.**
| | VR Art | VR Only | Control |
|---|---|---|---|
| Pre (centre) | Optimism | Calm & tranquil | The surgery |
| Post (centre) | Video & nature | Video & nature | The surgery |

- Before VR the centres are anticipation topics (optimism, calm, the operation). After VR, the two VR groups are centred on the video and nature; Control stays centred on the surgery.
- Control's "The surgery" is the strongest hub at both times (PageRank .26 pre, .20 post).

**Caveat.** Each panel rests on only 16-23 transitions. Tests of PageRank change per topic found nothing (smallest adjusted p = .37), and Jensen-Shannon tests of the graphs found no difference within or between conditions (all p >= .16, FDR >= .49). Treat the networks as a descriptive picture, not a tested effect.

---

## Slide 2 - Semantic space: everyone moves toward calm (`prepost_semantic_space.pdf`)

**What it shows.** Each thin line is one participant from their pre descriptors (open dot) to their post descriptors (filled dot) in the embedding space; large diamonds and arrows are the condition means. Axes are the two principal components that move most from pre to post (PC3, sign-flipped so post lies to the upper right, and PC1); they were picked for display, the tests use the full embedding. A `*` beside a diamond marks a significant pre-to-post shift; the PC beside it is the one that changes significantly within that condition. The two boxes give the p-values and the PCs.

**How.** Each participant's descriptors are averaged into one vector per timepoint. Paired permutation tests (pre/post swapped within participant) test the shift; condition x time differences test whether the shift differs; PCs are tested one by one and corrected over the first 10.

**What it shows.**
- **Overall shift:** semantic space p < .001, topic composition p = .001 (all 49 paired participants).
- **Within condition (semantic space, FDR over the three conditions):** VR Art p = .004, VR Only p = .004, Control p = .18. Only the two VR groups shift significantly.
- **Which direction:** PC1 runs from worries (low) to calm and feeling fine (high); PC3 from anticipation (low, optimism and expecting it to go well) to calm and video (high). Everyone moves up on PC1 (pooled adjusted p = .031) and along PC3 toward calm and video (pooled adjusted p = .001). PC3 is also significant within VR Art (adjusted p = .027) and VR Only (adjusted p = .016), not within Control.
- **Topics:** "Anxiety & fear" falls from 26% to 9% of descriptors overall, and "Video & nature" and "Feeling calmer" rise.

**Caveats.**
- The between-condition test of the change is **not** significant (semantic space p = .23; VR Only vs Control p = .069, FDR .21). "Shifts in VR but not in Control" is not evidence that VR changes more; Control has 15 participants and moves in the same direction.
- The map shows about 17% of the variance in two axes, and individual participants move in many directions.
- The PC end labels (worries / calm & fine; anticipation / calm & video) are my interpretation of where topics sit on each PC.

---

## Slide 3 - Semantic projection: relief on a distress-relief axis (`semantic_projection_prepost_change.pdf`)

**What it shows.** Each panel is one condition. Lines are participants' mean relief score before (open) and after (filled); the thick line is the model's change. Higher = more relief, lower = more distress.

**How.** Each descriptor is embedded and projected onto a distress-vs-relief axis, the same method as our earlier semantic projection analysis. The axis is the average direction from six distress anchor words to six relief anchor words, here in Portuguese translations because the descriptors are Portuguese. The main model is a descriptor-level mixed model with participant random intercept (`projection ~ time * condition`), with cluster-robust (CR2) inference and BH-FDR within each family of contrasts. Participant-level paired tests are a sensitivity analysis.

**What it shows.**
| | Change (post - pre) | 95% CI | p | FDR-adjusted p |
|---|---|---|---|---|
| VR Only | +0.142 | [0.045, 0.239] | .007 | .022 |
| VR Art | +0.106 | [0.006, 0.205] | .039 | .059 |
| Control | +0.054 | [-0.024, 0.132] | .158 | .158 |
| All conditions pooled | +0.101 | [0.051, 0.150] | < .001 | |

- Relief rises in every condition; after FDR adjustment only VR Only is significant.
- **The change does not differ significantly between conditions:** VR Art vs Control +0.052 (p = .38), VR Only vs Control +0.088 (p = .14), planned VR (Art + Only) vs Control +0.070 [-0.030, 0.169] (p = .16). Baseline and post-interview condition differences are also null.

**Caveat.** The mixed model weights each descriptor equally, so participants who gave more descriptors count more; this is why VR Only looks larger than the others. In the participant-level sensitivity analysis the three conditions are almost identical (+0.119, +0.112, +0.102; no difference in change, p = .97). The model choice is stated up front as the main model; the participant-level result is the conservative reading.

---

## Slide 4 - Semantic space before VR (`topic_network_pre/outputs/semantic_space_by_condition.pdf`)

**What it shows.** All pre-interview descriptors (small dots), each participant's average (large dots) and each condition's mean (diamond, with an ellipse for the 95% bootstrap region) in the embedding space of the descriptors. Axes are the first two principal components (PC1 12.5%, PC2 7.7% of the variance); no PC separates the conditions, so the plain leading PCs are used. Topic names sit at each topic's centre. The heat table gives the permutation p-values; the right box states that no PC separates the conditions.

**How.** 135 descriptors from 51 participants (VR Art 17, VR Only 18, Control 16; the small "unassigned" topic is left out) from the rater-consensus Q1 answers ("how do you feel now?"), embedded with Qwen3-Embedding. Group differences are tested by PERMANOVA on cosine distances (sentence level) and on participant averages, permuting participants' condition labels; spread is tested separately; each PC is tested and corrected over the first 10.

**What it shows.**
- **The three conditions overlap almost completely.** Descriptor-level PERMANOVA p = .45, participant-level p = .64, spread p = .89; no pair differs (participant-level pairs p >= .33, Holm >= .99; descriptor-level p >= .21, Holm >= .62).
- No PC separates the conditions (smallest FDR-adjusted p = .081, PC5).
- Together with the other pre-interview tests (topic composition, transitions, flow-graph JSD, relief projection), none significant after correction, this is a **clean baseline**: the groups were comparable before VR. (One nominal result: topic composition VR Art vs VR Only p = .029, Holm .088.)

**Caveat.** With 16-18 participants per group, only large baseline differences would be detected; these tests do not prove the groups were identical.

---

## Slide 5 - Semantic space after VR (`topic_network_post/outputs/semantic_space_by_condition.pdf`)

**What it shows.** The same kind of map for the **post-interview** answers: each small dot is a sentence, large dots are participants' averages, diamonds and ellipses are the condition means. Axes are PC4 (4.5% of the variance) and PC6 (3.0%), the two PCs that separate the conditions best, chosen for display; the tests use the full embedding. The left box gives the test p-values (VR Art vs Control is the one that stands out); the right box gives the two PCs with their end labels and each condition's mean on them.

**How.** All 324 sentences of the post interview (three questions: how do you feel now, what has changed, which aspects of care mattered) from 51 participants (17 per condition), embedded with Qwen3-Embedding. PERMANOVA on sentence-level cosine distances and on participant averages, with participants' condition labels permuted; Holm over the three pairs. PCs tested one by one, corrected (BH) over the first 10; the PC end labels come from where the topics sit on each PC and the sentences at its two ends.

**What it shows.**
- **VR Art differs from Control in what participants say:** sentence-level PERMANOVA p = .010 and participant-level p = .034 (Holm-adjusted). VR Only sits in between and is not reliably different from either group; the test across all three is a trend (p = .082 sentence level, .069 participant level). Spread does not differ (p = .89), so this is a shift in location, not in variability.
- **Two PCs carry it.** PC4 (FDR p = .003) runs from **Enjoyment** (film and nature) to **Worries (surgery, care)**, and VR Art > VR Only > Control on it; all three pairs differ after correction. PC6 (FDR p = .044) runs from **Relief** to **Worries (anxiety)**, with Control lowest (VR Art vs Control adjusted p = .015).
- In plain terms: VR Art participants talk more about the film and calm, Control participants more about the operation, worries and care.

**Caveats.**
- **Exploratory:** this was run after the topic-based tests came out empty, and the map axes were chosen because they separate the groups; the picture looks more separated than the tests warrant.
- **Not comparable to the pre-post analysis:** the post map uses all three interview questions (raw sentences), the pre and pre-post analyses use only the Q1 descriptors. Questions 2 and 3 are asked only after the intervention and invite talk about the film and about care, which may partly explain why the groups differ here.
- **A post-only difference is not a change:** the descriptor-based pre-to-post tests show no reliable difference in the size of change between conditions (slide 2), so this result should be read as "the groups talk about different things afterwards", not "VR changed people more".

---

## Overall take-home and limitations

1. **From pre to post, language moves toward calm and away from worry in all conditions** (pooled tests strongly significant). Semantic space and the relief projection agree.
2. **The VR groups show more consistent shifts within condition, but no test shows that the change is larger in VR than in Control.** The interaction tests (semantic space, topic composition, flow graphs, relief projection) are all non-significant. Before VR the groups do not differ (slide 4); the full post-interview answers do show VR Art talking about different things than Control (slide 5), but that is an exploratory, post-only result on a different data set.
3. **Small samples:** 15-17 participants per condition, 2-3 descriptors per participant, 16-23 transitions per flow graph. Only large differences between conditions would be detectable, and the graph results are descriptive.
4. **Exploratory choices:** map axes and PC labels were picked or written after seeing the data; the Portuguese anchor translations are mine; the topic model is fitted on pre and post together.
5. **Not causal:** Control moves in the same direction, so the pre-post change alone does not identify an effect of VR; other things change between the two interviews (for example, time spent in the hospital) and are not separable here.
