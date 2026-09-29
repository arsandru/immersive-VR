# Topic network analysis of the pre-interview descriptors

The analysis of `topic_network_post/` (topic flow graphs, betweenness / PageRank / semantic centrality, PageRank and path tests, global distribution tests, embedding-space content tests, PC differences, core networks, semantic-space map) run on the **pre-interview Q1 descriptors** (`data/consensus_pre_q1.csv`, one unit per rater-consensus descriptor). The scripts are copies of the post-analysis scripts with three adaptations:

* **Input.** `prepare_pre_data.py` takes the pre units of `pre_post/outputs/prepost_units_with_topics.csv`: topics come from the one BERTopic model fitted on pre + post descriptors together (so a topic means the same here and in `pre_post/`); labels are in `pre_topic_labels.py` (the short labels written for `pre_post/`, three of which - Calm & tranquil, Feeling calmer, Relaxed - overlap in meaning). Embeddings are the cached Qwen3 vectors of `pre_post/`. 135 descriptors (outlier topic dropped), 51 participants (VR Art 17, VR Only 18, Control 16), 11 topics.
* **One flow scope.** Each participant has a single Q1 list, so the "within response" and "across questions" flow scopes of the post analysis coincide; only `within_response` is run and all figures have one scope.
* **No PC survives the FDR test** (smallest adjusted p = .081, PC5), so `pc_end_evidence.py` collects nothing, the semantic-space map uses PC1 and PC2, and its PC inset says so.

Statistics are unchanged from `topic_network_post/` (pairs are Holm-adjusted here, as there; the newer `pre_post/` analyses use BH-FDR).

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
* The conditions do not differ before the intervention: embedding-space PERMANOVA p = .45 (sentence-level) / .64 (participant-level), dispersion p = .89; global composition p = .18 (GJSD) / .53 (PERMANOVA), transitions p = .62, stationary distribution p = .33; graph JSD between path distributions p = .42, pairs p >= .30 (Holm >= .9). One nominal result: topic composition VR Art vs VR Only p = .029 (Holm .088).
* Most central topic per condition (PageRank): VR Art Optimism (.23), VR Only Calm & tranquil (.17), Control The surgery (.26; higher than the maximum expected under shuffled order, p = .047, the only condition where the central topic is more central than chance).
* Per-topic PageRank vs shuffled order: several topics are less central than their frequency predicts (Anxiety & fear in VR Art and VR Only) or more (Calm & tranquil in VR Only), FDR < .05; each graph rests on only 16-23 transitions, so these are descriptive.

Outputs are in `outputs/` (same file names as `topic_network_post/outputs/`, without the across-questions files).
