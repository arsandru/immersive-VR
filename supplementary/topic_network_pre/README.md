# Topic network analysis of the pre-interview descriptors

The analysis of `topic_network_post_allquestions/` (the original analysis of the full post interview; `topic_network_post/` is its Q1-descriptor counterpart to this folder) (topic flow graphs, betweenness / PageRank / semantic centrality, PageRank and path tests, global distribution tests, embedding-space content tests, PC differences, core networks, semantic-space map) run on the **pre-interview Q1 descriptors** (`data/consensus_pre_q1.csv`, one unit per rater-consensus descriptor). The scripts are copies of the post-analysis scripts with three adaptations:

* **Input.** `prepare_pre_data.py` takes `pre_post/outputs/pre_units_with_topics.csv`: topics come from a BERTopic model fitted on the pre descriptors only (`pre_post/fit_timepoint_topics.py`), so no post-interview information enters them; labels come from `pre_post/timepoint_topic_labels.py`, the single source shared with `pre_post/` and `topic_network_post/` (interpretations written after reading sample descriptors; "Calm & tranquil", "Calm but a bit anxious" and "Relaxed" overlap in meaning). Embeddings are the cached Qwen3 vectors of `pre_post/`. 135 descriptors (unassigned topic dropped), 47 participants (VR Art 15, VR Only 17, Control 15), 10 topics.
* **One flow scope.** Each participant has a single Q1 list, so the "within response" and "across questions" flow scopes of the post analysis coincide; only `within_response` is run and all figures have one scope.
* **No PC survives the FDR test** (smallest adjusted p = .081, PC5), so `pc_end_evidence.py` collects nothing, the semantic-space map uses PC1 and PC2, and its PC inset says so.

Statistics are unchanged from `topic_network_post_allquestions/` and `topic_network_post/` (pairs are Holm-adjusted here, as there; the newer `pre_post/` analyses use BH-FDR).

## Run order (from this folder)
```
python3 prepare_pre_data.py
python3 topic_network_analysis.py            # ~3 min: graphs, centralities, bootstrap CIs, condition tests
python3 pagerank_within_condition_tests.py
python3 path_distribution_tests.py
python3 global_distribution_tests.py
python3 path_graph_jsd_tests.py              # ~1.5 min
python3 embedding_space_tests.py
python3 pc_differences.py
python3 pc_end_evidence.py                   # exits early: no PC passes FDR
python3 plot_topic_network.py
python3 plot_semantic_space.py
```

## Results (pre interviews, condition comparison before VR)
* The conditions do not differ before the intervention: embedding-space PERMANOVA p = .65 (descriptor level) / .63 (participant level), dispersion p = .32; global composition p = .52 (GJSD) / .69 (PERMANOVA), transitions p = .29, stationary distribution p = .69; flow-graph JSD between path distributions p = .37, pairs p >= .28 (Holm >= .85). No pairwise test is significant after Holm; no PC separates the conditions (smallest adjusted p = .88).
* Most central topic per condition (PageRank): VR Art Calm & tranquil (.17), VR Only Physical discomfort (.23), Control Expecting it to go well (.17). None is more central than expected under shuffled order (p >= .94), and no topic is more or less central than its frequency predicts after FDR.
* Each graph rests on only 15-28 transitions, so the graph results are descriptive.

Outputs are in `outputs/` (same file names as `topic_network_post/outputs/`, without the across-questions files).

Parameters, clusters and labels are the same as in the pre row of `pre_post/outputs/core_network_pre_vs_post.pdf` and are the same for pre and post (both min cluster size 5, min samples 2; see `pre_post/timepoint_topic_labels.py`); `pre_post/check_topic_consistency.py` verifies this.
