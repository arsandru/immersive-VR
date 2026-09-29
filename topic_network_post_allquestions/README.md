> **Renamed.** This is the original analysis of the full post interview (three questions, 324 sentences, its own topic model). It was called `topic_network_post/`; that name now belongs to the analysis of the post Q1 descriptors, which mirrors `topic_network_pre/` and uses the same clusters, labels and parameters as `pre_post/`.

# Topic network and semantic betweenness centrality

Builds networks over the sentence-level BERTopic topics from `all_questions` and
computes (probabilistic) semantic betweenness centrality per condition, after
Lande et al., "Probabilistic Semantic Betweenness Centrality in Cognitive Networks".

Input: `all_questions/outputs/combined_questions_sentence_topic_assignments.csv`
(the outlier topic -1 is dropped; use `--include-outlier` to keep it).

## Graphs (built separately for VR Art, VR Only, Control)
- **Flow, within response** - directed edge A->B when a sentence in topic A is followed by one in topic B inside the same response.
- **Flow, across questions** - same, but along each participant's whole interview (Q1 -> Q2 -> Q3, ordered by sentence). Includes the within-response transitions plus the Q-to-Q boundaries.
- **Semantic** - undirected k-nearest-neighbour graph (default k=3) on topic centroids of Qwen3-Embedding-0.6B sentence embeddings, centroids computed from the condition's own sentences.

Self transitions (A->A) are counted but do not enter the graph. Edge probability
`P_ij` = count(i->j) / outgoing count of i (self transitions excluded).

## Measures (normalised by (K-1)(K-2))
| measure | optimal path | note |
|---|---|---|
| `SBC_hop` | fewest hops | classical betweenness |
| `PSBC_hop` | fewest hops, each path weighted by prod(P_ij) | paper's PSBC; differs from SBC only through ties in hop length |
| `SBC_prob` | flow graph: most probable path (length -log P) | |
| `SBC_sim` | semantic graph: most similar path (length 1 - cosine) | |

Semantic-graph transition probabilities follow the paper: w_ij = Score(j) * sim_ij / d_in(j) with Score = topic share in the condition (the similarity factor is an addition so that edge strength matters). The paper's iterative Score update is not implemented.

## "What is central" measures
| measure | graph | definition |
|---|---|---|
| `PageRank` | flow (both scopes) | stationary weight of the transition graph, damping 0.85, weights P_ij, sums to 1 (uniform = 1/K). High = the narrative keeps landing on the topic. |
| `SemCentrality` | semantic | prevalence-weighted mean cosine similarity of a topic to the other topics' centroids. High = topic sits at the semantic centre of what the condition says. Topics absent from a condition use the all-participant centroid (weight 0). |

Main figures: `outputs/central_topics_by_condition.*` (dot plot) and `outputs/core_network_{within_response,across_questions}.*` (network per condition: most central topic in the middle, next three on an inner ring, the rest on an outer ring; node size/colour = PageRank; arrows = transitions seen at least twice).

## Uncertainty and tests
- 95% CIs: participant bootstrap (`--n-boot`, default 500).
- Condition differences: permutation of participant condition labels (`--n-perm`, default 1000), Benjamini-Hochberg FDR within each graph/scope/measure.

## Run
```
python3 topic_network_post_allquestions/topic_network_analysis.py     # ~3 min
python3 topic_network_post_allquestions/plot_topic_network.py
```
Outputs in `outputs/`: `centrality_by_condition.csv`, `centrality_condition_tests.csv`, `network_edges.csv`, `network_descriptives.csv`, `run_summary.json`, cached embeddings, and `network_*` / `centrality_*` figures (pdf, png).

## Caveats
About 100 sentences per condition (17 participants each), so many topic-to-topic transitions are observed once or never. Treat results as exploratory: CIs are wide and no condition difference survives FDR correction.

## Within-condition PageRank tests
`pagerank_within_condition_tests.py` (run before the plots; seconds):
1. **PageRank vs. chance** (`pagerank_within_condition_topic_tests.csv`, `pagerank_vs_null_*`): every topic's PageRank vs a null where sentence order is shuffled inside each sequence (frequencies kept, transition structure destroyed); two-sided permutation p, BH-corrected over topics. Shows which topics are more or less central than their frequency alone predicts.
2. **Contributions to the most central topic** (`pagerank_top_topic_contributions.csv`, `pagerank_contributions_*`): PageRank decomposes exactly as PR_c = baseline + sum_j 0.85 * PR_j * P_jc, so each source topic j has a contribution. Sources with an observed transition into c are tested one-sided against the same null (BH over sources); bootstrap 95% CIs over participants. Baseline = teleport + dangling redistribution.

## Paths toward the central topics, compared across conditions
`path_distribution_tests.py` (seconds). Targets = topics that are most central (PageRank) in at least one condition, per flow scope (within responses: Calm, Personal effects, Surgery worries; across questions: Calm, Caring staff). For each target and condition, paths start from the observed opening-topic distribution and end at the first arrival at the target within 3 steps; path probability = product of transition probabilities (additive smoothing 0.1 per off-diagonal cell; 0.02 and 0.5 as sensitivity). Reported: reach probability, entry routes (last topic before the target), top paths. Conditions are compared with participant-label permutation tests (reach, per-route entry mass with BH, Jensen-Shannon distance between route distributions). Outputs: `path_*.csv`, `path_distribution_{scope}.*`.

## Global distribution tests
`global_distribution_tests.py` (seconds). No central topic is singled out; participants are permuted across conditions (5000 permutations). (A) Topic composition: pooled-sentence generalised JSD between the three conditions' topic distributions, and a participant-level PERMANOVA on Hellinger-transformed topic proportions; per-topic contributions to the GJSD. (B) Transition structure per scope: source-weighted GJSD of the conditions' outgoing-transition distributions (conditions on the source topic, so it is independent of topic frequency), per-source contributions, and GJSD between the smoothed stationary (PageRank) distributions. Pairwise GJSD tests with Holm correction. Outputs: `global_distribution_tests.csv`, `global_distribution_contributions.csv`, `global_distribution_tests.*`, `global_distribution_contributions.*`.

## JSD test on path distributions between condition graphs
`path_graph_jsd_tests.py` (~90 s). Each condition's flow graph is a Markov chain (smoothing 0.1 per off-diagonal cell; 0.02 and 0.5 as sensitivity) that assigns a probability to every L-step path, Prob = pi0(i0) * prod P. The Jensen-Shannon divergence (bits) between two conditions' distributions over all 3-step paths (main; 2 steps as sensitivity) measures how differently their graphs route the walk; omnibus = generalised JSD of the three. Opening distribution: pooled over all participants (so only the graphs differ; main) or each condition's own. Participant-label permutation test (5000), Holm over the three pairs. Outputs: `path_graph_jsd_tests.csv`, `path_graph_jsd_top_paths.csv` (paths contributing most to the divergence), `path_graph_jsd_tests.*`, `path_graph_jsd_top_paths.*`.

## Content tests in the sentence-embedding space (no topic labels)
`embedding_space_tests.py` (~10 s) and `plot_semantic_space.py`. Uses the cached Qwen3 sentence embeddings (`outputs/sentence_embeddings.npz`). Do the conditions' sentences sit in different places in meaning-space? Participant labels are permuted (5000; sentences stay with their participant), Holm over the three pairs: PERMANOVA on sentence-level cosine distances, PERMANOVA on participant centroids, and dispersion of participant centroids. Outputs: `embedding_space_tests.csv`, `semantic_space_by_condition.*` (PCA map with participants and condition means with bootstrap 95% regions, next to a heat table of the tests), `embedding_space_tests.*` (p-value table).

## Which PCs differ and what they point toward
`pc_differences.py` (~1 min) and `pc_differences_by_condition.*`. Uses the same PCs as the map. First 10 PCs (fixed in advance): permutation ANOVA on participants' PC scores (participant labels shuffled, 10000), Benjamini-Hochberg over the 10 PCs, eta^2 as effect size; pairwise tests (Holm over three pairs) only for PCs that survive. A PC's meaning is read from where each topic's centroid sits on it (`pc_topic_loadings.csv`) and from the sentences at its two ends (`pc_extreme_sentences.csv`). PC signs are arbitrary. Outputs: `pc_condition_tests.csv`, `pc_topic_loadings.csv`, `pc_extreme_sentences.csv`, `pc_differences_by_condition.*`.

## What sits at the ends of the significant PCs
`pc_end_evidence.py` (~1 min) collects three checks per end (top vs bottom quartile of sentence scores) for the PCs that pass the FDR test: (1) distinctive words by weighted log-odds with an informative Dirichlet prior, counted as supported only with |z| > 1.96, >= 3 participants and >= 90% stability under participant-cluster bootstrap; (2) topic positions on the PC with participant-cluster bootstrap 95% CIs; (3) which conditions the end sentences come from. With ~80 sentences per end no word reaches the strict threshold, so the words only corroborate direction; the topic positions carry the evidence. Short labels (`PC_END_LABELS` in the script: PC4 Enjoyment vs Worries (surgery, care), PC6 Relief vs Worries (anxiety)) are an interpretation written after inspecting this evidence and are stored with it in `pc_end_labels.csv`; `pc_end_words.csv`, `pc_end_topic_positions.csv`, `pc_end_condition_shares.csv` hold the full tables. A keyword check (fixed stem lists written after seeing the evidence, so exploratory) is stored in the same file: share of end sentences containing enjoyment / relief / worry stems. The map inset uses these labels.

Map axes: `semantic_space_by_condition` now uses the two PCs that separate the conditions most (largest eta^2 among those passing the FDR test: PC4 and PC6), picked for display after seeing the data; the tests use the full embedding and do not depend on the axes.
