# Topic network analysis of the post-interview descriptors

The analysis of `topic_network_pre/` (topic flow graphs, betweenness / PageRank / semantic centrality, PageRank and path tests, global distribution tests, embedding-space content tests, PC differences, core networks, semantic-space map) run on the **post-interview Q1 descriptors** (`data/consensus_post_q1.csv`, one unit per rater-consensus descriptor). It is the mirror image of `topic_network_pre/`: same scripts, same statistics, same topic-model parameters, and the clusters and labels are the very ones shown in the post row of `pre_post/outputs/core_network_pre_vs_post.pdf`.

* **Input.** `prepare_post_data.py` reads `pre_post/outputs/post_units_with_topics.csv`: the clusters of a BERTopic model fitted on the post descriptors only (`pre_post/fit_timepoint_topics.py`), with the parameters and labels of `pre_post/timepoint_topic_labels.py` (single source shared with `pre_post/` and `topic_network_pre/`; `pre_post/check_topic_consistency.py` verifies clusters, labels, parameters and the central topic of every panel). Labels are interpretations written after reading sample descriptors. 109 descriptors (unassigned topic dropped), 48 participants (16 per condition), 10 topics.
* **One flow scope.** Each participant has one Q1 list, so the within-response and across-question scopes coincide; only `within_response` is run.
* **The earlier analysis of the full post interview** (three questions, 324 sentences, its own topic model with 12 topics) is kept unchanged in `topic_network_post_allquestions/`.
* No PC survives the FDR test (smallest adjusted p = .446), so the semantic-space map uses PC1 and PC2 and `pc_end_evidence.py` exits early.

Statistics are those of `topic_network_pre/` (pairs Holm-adjusted; the newer `pre_post/` analyses use BH-FDR).

## Run order (from this folder)
```
python3 prepare_post_data.py
python3 topic_network_analysis.py
python3 pagerank_within_condition_tests.py
python3 path_distribution_tests.py
python3 global_distribution_tests.py
python3 path_graph_jsd_tests.py
python3 embedding_space_tests.py
python3 pc_differences.py
python3 pc_end_evidence.py                   # exits early: no PC passes FDR
python3 plot_topic_network.py
python3 plot_semantic_space.py
```

## Results (post interviews, condition comparison after VR)
* Embedding space: descriptor-level PERMANOVA p = .044 across the three conditions; VR Art vs Control p = .006 (Holm .017), VR Only vs Control p = .12 (Holm .23), VR Art vs VR Only p = .59. Participant-level PERMANOVA p = .10 (VR Art vs Control p = .035, Holm .105). Spread does not differ (p = .995). No PC separates the conditions.
* Topic composition (post-only topics): GJSD p = .037, PERMANOVA p = .019; VR Art vs Control p = .017 (Holm .051; PERMANOVA .012, Holm .035), VR Only vs Control p = .031 (Holm .062). VR Art descriptors fall more often in "Feeling calmer" (26% vs 8% in Control) and "Video & nature" (24% vs 0%), Control descriptors in "Anxiety & fear" (25% vs 5%) and "Surgery & coping" (25% vs 5%); no single topic survives FDR (Video & nature p = .012, FDR .12).
* Transitions p = .76, stationary distribution p = .13, flow-graph JSD p = .76 (pairs p >= .49): the topic flow does not differ.
* Most central topic per condition (PageRank): VR Art Feeling calmer (.24), VR Only Video & nature (.17), Control Anxiety & fear (.23); none more central than expected under shuffled order (p >= .52); no topic differs from its frequency after FDR. Each graph rests on 15-16 transitions, so the graph results are descriptive.
