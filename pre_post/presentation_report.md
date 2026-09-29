# Pre vs post VR: presentation report

**Data.** Rater-consensus descriptors of Q1 ("how do you feel now?") from the pre and post interviews: 150 pre and 125 post descriptors, 49 participants with both interviews (VR Art 17, VR Only 17, Control 15). Portuguese text, embedded with Qwen3-Embedding-0.6B. Conditions do not differ before the intervention (baseline: topic composition p = .52, semantic space p = .65, flow graphs p = .37; the composition and flow-graph baselines use the shared-topic model described below). Multiple comparisons: Benjamini-Hochberg FDR throughout.

**Topics.** Topics are extracted separately for each timepoint: one BERTopic model on the pre descriptors only and one on the post descriptors only (10 topics each, identical parameters), so neither timepoint's topics are shaped by the other interview. The clusters, labels and parameters are the same in the core networks (slide 1), `topic_network_pre` (slide 4) and `topic_network_post` (slide 5); `check_topic_consistency.py` verifies it. A few tests compare topic categories across timepoints (topic composition change, per-topic PageRank change, flow-graph JSD pre vs post), which needs one common set of topics; these still use one model fitted on pre and post descriptors together ("shared-topic model", 11 topics) and are reported as such. All semantic-space and relief-projection results do not use topics.

**Sources.** Slide 6 compares the two topic sets directly (`topic_distance`). Slides 1-3 use the Q1 descriptors at both timepoints (`pre_post/`, `semantic_projection_prepost/`); slide 4 uses the pre descriptors (`topic_network_pre/`); slide 5 uses the post-interview Q1 descriptors (`topic_network_post/`); a supplementary analysis of the full post interview (three questions) is in `topic_network_post_allquestions/`.

**One-sentence message.** From pre to post, everyone's language moves from worry and anticipation toward calm; VR Art and VR Only shift significantly, Control does not reach significance, but the size of the change does not reliably differ between conditions.

---

## Slide 1 - Core networks: what participants' talk revolves around (`core_network_pre_vs_post.pdf`)

**What it shows.** One network per condition (columns) before (top row) and after (bottom row). Nodes are the 10 topics extracted from that timepoint's own descriptors, so the topics in the top and bottom rows are different sets and are not matched one to one; arrows are observed transitions between consecutive descriptors in one participant's answer (thicker = more often). Node size and colour are PageRank, the share of "attention" a topic collects when following the transitions; the same scale is used in all six panels. The most central topic sits in the middle, the next three on an inner ring, the rest outside.

**How.** One BERTopic model per timepoint (pre-only and post-only descriptors), with the same settings. PageRank (damping 0.85) is computed per condition and timepoint from that condition's transition counts.

**What it shows.** Most central topic (PageRank in brackets):
| | VR Art | VR Only | Control |
|---|---|---|---|
| Pre (centre) | Calm & tranquil (.17) | Physical discomfort (.23) | Expecting it to go well (.17) |
| Post (centre) | Feeling calmer (.24) | Video & nature (.16, tied with Relaxed) | Anxiety & fear (.23, tied with Surgery & coping) |

- Before VR the centres are calm, anticipation and, in VR Only, physical discomfort (hunger, thirst, discomfort while waiting).
- After VR, VR Art is centred on feeling calmer and VR Only on the video, while Control's centre is anxiety and the surgery.
- The pre topic set contains topics the post set does not (for example "Physical discomfort" and "Trusting the team") and the post set has its own ("Video & nature", "Comfort & peace"), which is the point of extracting them separately.

**Caveat.** Each panel rests on only 15-28 transitions, and a centre can be a near-tie (Control post: Anxiety & fear and Surgery & coping both .23). In the pre data no condition's central topic is more central than expected under shuffled order (p >= .94), and the topic-based comparison tests across timepoints (shared-topic model) found no PageRank change per topic (smallest adjusted p = .37) and no Jensen-Shannon difference within or between conditions (all p >= .16, FDR >= .49). Treat the networks as a descriptive picture, not a tested effect.

---

## Slide 2 - Semantic space: everyone moves toward calm (`prepost_semantic_space.pdf`)

**What it shows.** Each thin line is one participant from their pre descriptors (open dot) to their post descriptors (filled dot) in the embedding space; large diamonds and arrows are the condition means. Axes are the two principal components that move most from pre to post (PC3, sign-flipped so post lies to the upper right, and PC1); they were picked for display, the tests use the full embedding. A `*` beside a diamond marks a significant pre-to-post shift; the PC beside it is the one that changes significantly within that condition. The two boxes give the p-values and the PCs.

**How.** Each participant's descriptors are averaged into one vector per timepoint. Paired permutation tests (pre/post swapped within participant) test the shift; condition x time differences test whether the shift differs; PCs are tested one by one and corrected over the first 10.

**What it shows.**
- **Overall shift:** semantic space p < .001 (all 49 paired participants); topic composition p = .001 (shared-topic model).
- **Within condition (semantic space, FDR over the three conditions):** VR Art p = .004, VR Only p = .004, Control p = .18. Only the two VR groups shift significantly.
- **Which direction:** PC1 runs from worries (low) to calm and feeling fine (high); PC3 from anticipation (low, optimism and expecting it to go well) to calm and video (high). Everyone moves up on PC1 (pooled adjusted p = .031) and along PC3 toward calm and video (pooled adjusted p = .001). PC3 is also significant within VR Art (adjusted p = .027) and VR Only (adjusted p = .016), not within Control.
- **Topics (shared-topic model):** "Anxiety & fear" falls from 26% to 9% of descriptors overall, and "Video & nature" and "Feeling calmer" rise.

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

**What it shows.** All pre-interview descriptors (small dots), each participant's average (large dots) and each condition's mean (diamond, with an ellipse for the 95% bootstrap region) in the embedding space of the descriptors. Axes are the first two principal components; no PC separates the conditions, so the plain leading PCs are used. Topic names sit at each topic's centre; the topics are extracted from the pre descriptors only. The heat table gives the permutation p-values; the right box states that no PC separates the conditions.

**How.** 135 descriptors from 47 participants (VR Art 15, VR Only 17, Control 15) from the rater-consensus Q1 answers ("how do you feel now?"), after leaving out the small "unassigned" topic of the pre-only topic model, embedded with Qwen3-Embedding. Group differences are tested by PERMANOVA on cosine distances (descriptor level) and on participant averages, permuting participants' condition labels; spread is tested separately; each PC is tested and corrected over the first 10.

**What it shows.**
- **The three conditions overlap almost completely.** Descriptor-level PERMANOVA p = .65, participant-level p = .63, spread p = .32; no pair differs (participant-level pairs p >= .40; descriptor-level p >= .19, Holm >= .58).
- No PC separates the conditions (smallest FDR-adjusted p = .88).
- The other pre-interview tests on the pre-only topics agree: topic composition p = .52 (pooled JSD) and .69 (PERMANOVA), transitions p = .29, stationary distribution p = .69, flow-graph JSD p = .37, and no topic is more or less central than its frequency predicts. This is a **clean baseline**: the groups were comparable before VR.

**Caveat.** With 15-17 participants per group, only large baseline differences would be detected; these tests do not prove the groups were identical.

---

## Slide 5 - Semantic space after VR (`topic_network_post/outputs/semantic_space_by_condition.pdf`)

**What it shows.** The same kind of map as slide 4 for the **post-interview** Q1 descriptors: small dots are descriptors, large dots participants' averages, diamonds and ellipses the condition means (95% bootstrap region). Axes are PC1 (13.1% of the variance) and PC2 (7.5%); no PC separates the conditions. Topic names sit at each topic's centre; the topics are extracted from the post descriptors only, with exactly the same procedure, parameters and labels as for the pre interview. The heat table gives the test p-values.

**How.** 109 descriptors from 48 participants (16 per condition; the unassigned topic left out) from the rater-consensus Q1 answers of the post interview, embedded with Qwen3-Embedding. PERMANOVA on cosine distances (descriptor level) and on participant averages, with participants' condition labels permuted, Holm over the three pairs; spread tested separately; PCs tested one by one and corrected over the first 10.

**What it shows.**
- **VR Art differs from Control in what participants say:** descriptor-level PERMANOVA p = .006 (Holm .017); across all three conditions p = .044. VR Only sits in between (vs Control p = .12; Holm .23). At participant level the picture is weaker (VR Art vs Control p = .035, Holm .105; all three p = .10). Spread does not differ (p = .995).
- **The topics say the same thing:** topic composition differs (p = .037 pooled, PERMANOVA p = .019), driven by VR Art vs Control (p = .017, Holm .051; PERMANOVA Holm .035). VR Art descriptors fall more often in "Feeling calmer" (26% vs 8% in Control) and "Video & nature" (24% vs 0%); Control descriptors more often in "Anxiety & fear" (25% vs 5%) and "Surgery & coping" (25% vs 5%). No single topic survives FDR.
- **Topic flow does not differ:** transitions p = .76, stationary distribution p = .13, flow-graph JSD p = .76. Most central topics: VR Art Feeling calmer (.24), VR Only Video & nature (.17), Control Anxiety & fear (.23).

**Caveats.**
- This is a **post-only comparison, not a change**: the pre-to-post tests show no reliable difference in the size of change between conditions (slide 2). Read it as "the groups talk about different things afterwards".
- With 16 per group, participant-level tests are weaker than descriptor-level tests (participants with more descriptors count more at descriptor level).
- **Supplementary analysis of the whole post interview** (all three questions, 324 sentences, its own topic model with 12 topics; `topic_network_post_allquestions/`): the same pattern, VR Art differs from Control (sentence-level PERMANOVA Holm .010, participant-level Holm .034) along two PCs, "Enjoyment vs Worries (surgery, care)" (PC4) and "Relief vs Worries (anxiety)" (PC6); it uses questions 2 and 3 that are asked only after the intervention, and its axes were chosen for display, so it is exploratory.

---

## Slide 6 - How far are the pre topics from the post topics? (`topic_distance.pdf`; detail in `topic_distance_detail.pdf`)

**What it shows.** Left (within): the topic-set distance S, the size-weighted average distance from each topic to its nearest counterpart in the other timepoint, for all participants and for each condition, against the distance expected if pre and post made no difference (grey bar: up to the 95th percentile of chance; tick: chance mean). Right (between): the difference in S between two conditions, against the differences obtained when condition labels are shuffled. The full picture of every pre-post topic pair, and the two descriptor-to-topic distances, is in `topic_distance_detail.pdf`.

**How.** Topics are extracted separately at each timepoint, so they are compared through their centres in the embedding space. The comparison with chance swaps the pre and post labels of each participant's descriptors at random (1000 times) and refits both topic models each time.

**What it shows.**
- **Most topics have a close counterpart:** anxious and nervous talk before matches anxiety and fear after (distance .04), calm and tranquil matches feeling calmer (.05), feeling well matches feeling well (.04).
- **Overall the post topics are further from the pre topics than chance** (S = .090 against a chance mean of .067, p = .020; matched-topic distance M = .147 against .098, p = .017): the topic structure changes, mainly in a few places.
- **What is new or gone:** the pre topic "Physical discomfort" (hunger, thirst) has no close post counterpart (.20, just above the chance 95th percentile); the post topics "Video & nature" (.18) and "Emotion & unease" (.18) have no close pre counterpart (below the chance threshold).
- **Within each condition** (topic-set distance built from the condition's own descriptors, against its own chance level): VR Art p = .002 (FDR .006), VR Only p = .051 (FDR .076), Control p = .124 (FDR .124). Only VR Art's topic structure moves more than chance after correction; Control's does not, and VR Only sits at the edge.
- **Between conditions there is no reliable difference:** VR Art vs Control +.048 (p = .098, FDR .29), VR Only vs Control +.035 (p = .21, FDR .32), VR Art vs VR Only p = .60; the two descriptor-to-topic distances also do not differ (p >= .074, FDR >= .22). As before, "significant in one group but not in another" is not evidence that the groups differ.
- **Sensitivity** (dropping participants flagged for review, topics refitted): the overall set distance holds (p = .016), but the matched-topic distance does not (p = .30), and the within-condition pattern changes (VR Art and Control significant, VR Only p = .056), so the per-condition pattern is unstable.

**Caveat.** The topics are extracted from small samples (about 125-150 descriptors per timepoint), so a single topic boundary can move; the test is on the whole topic set, not on individual topics. This is a change in the structure of what is said, not evidence that VR caused it (Control changes too, and conditions do not differ).

---

## Overall take-home and limitations

1. **From pre to post, language moves toward calm and away from worry in all conditions** (pooled tests strongly significant). Semantic space and the relief projection agree.
2. **The VR groups show more consistent shifts within condition, but no test shows that the change is larger in VR than in Control.** The interaction tests (semantic space, topic composition, flow graphs, relief projection) are all non-significant. Before VR the groups do not differ (slide 4); the post-interview descriptors do show VR Art talking about different things than Control (slide 5, also in the full post interview), but that is a post-only comparison; the size of the pre-to-post change does not differ between conditions.
3. **Small samples:** 15-17 participants per condition, 2-3 descriptors per participant, 16-23 transitions per flow graph. Only large differences between conditions would be detectable, and the graph results are descriptive.
4. **Exploratory choices:** map axes and PC labels were picked or written after seeing the data; the Portuguese anchor translations are mine; the topic model is fitted on pre and post together.
5. **Not causal:** Control moves in the same direction, so the pre-post change alone does not identify an effect of VR; other things change between the two interviews (for example, time spent in the hospital) and are not separable here.
