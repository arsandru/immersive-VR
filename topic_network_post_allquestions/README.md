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

## Nearest-neighbour condition-mixing test (exploratory, no dimension reduction)
`knn_mixing_test.py` (~5 s): `knn_mixing_tests.csv`, `knn_mixing_tests.*`. For every sentence, its k = 10 nearest sentences by cosine similarity in the full Qwen3 embedding, excluding the same participant's sentences; score = share of those neighbours with the sentence's own condition, averaged within participant and then across participants. Omnibus over the three conditions; pairs use only the two conditions' sentences (neighbours drawn from the pair). Null: participant labels shuffled across participants, group sizes kept, neighbours fixed (10000 permutations, one-sided, Holm over the three pairs). Excess = observed minus permutation mean. Sensitivity: k = 5 and 20, sentence-weighted average. Per-condition rows are descriptive (BH). k = 10 and participant weighting were fixed before the final run.

Result (k = 10): VR Art vs Control 53.5% same-condition neighbours against 48.5% expected by chance (excess +5.0 points), p = 0.007, Holm 0.022; VR Art vs VR Only and VR Only vs Control at chance (Holm 0.96); omnibus p = 0.074. Holm for VR Art vs Control: 0.040 (k = 5), 0.029 (k = 20), 0.12 (sentence-weighted, p = 0.040). Same pattern as the PERMANOVA, from a local statistic rather than a variance decomposition; both use the same embedding, so this is not independent evidence. The effect is small.

## Which PCs differ and what they point toward
`pc_differences.py` (~1 min) and `pc_differences_by_condition.*`. Uses the same PCs as the map. First 10 PCs (fixed in advance): permutation ANOVA on participants' PC scores (participant labels shuffled, 10000), Benjamini-Hochberg over the 10 PCs, eta^2 as effect size; pairwise tests (Holm over three pairs) only for PCs that survive. A PC's meaning is read from sentences (typical sentences and word association, see "Reading the PCs from sentences alone" below; labels in `pc_end_labels.csv`); no topic model is involved. The five sentences at each end are listed in `pc_extreme_sentences.csv`. In `pc_differences_by_condition.*` only the PCs that pass FDR carry a label, and the matrix shows words (from `pc_sentence_words.csv`: for each of the first 10 PCs, sentences containing the word minus those without, in SD of the PC; dot = bootstrap CI excludes 0), not topics. PC signs are arbitrary. Outputs: `pc_condition_tests.csv`, `pc_extreme_sentences.csv`, `pc_differences_by_condition.*`.

## What sits at the ends of the significant PCs
`pc_end_evidence.py` (~1 min) collects checks per end (top vs bottom quartile of sentence scores) for the PCs that pass the FDR test: (1) distinctive words by weighted log-odds with an informative Dirichlet prior, counted as supported only with |z| > 1.96, >= 3 participants and >= 90% stability under participant-cluster bootstrap; (2) which conditions the end sentences come from; (3) a keyword check. With ~80 sentences per end no word reaches the strict threshold, so the words only corroborate direction; the typical sentences and the word association of `pc_sentence_interpretation.py` carry the interpretation. No topic model is used. Short labels (`PC_END_LABELS` in the script: PC4 Experience & senses vs Procedure & care; PC6 Calm, relaxed vs Anxiety, fear (wording)) are an interpretation written after inspecting this evidence and are stored with it in `pc_end_labels.csv`. They were revised after review: the earlier PC4 "Enjoyment" label rested on enjoyment words in only 16% of high-end sentences, and the earlier PC6 labels ("Relief" / "Worries (anxiety)") contradicted the words and sentences at its ends, so PC6 was relabelled again once grammatical and discourse words had been removed from the word table; `pc_end_words.csv` and `pc_end_condition_shares.csv` hold the full tables. The earlier topic-position and topic-loading tables (`pc_end_topic_positions.csv`, `pc_topic_loadings.csv`) and the topic labels on the map were removed. A keyword check (fixed stem lists written after seeing the evidence, so exploratory) is stored in the same file: share of end sentences containing enjoyment / relief / worry stems. The map inset and the PC axis ends use these labels. The representative words printed next to each label in the figures are the (up to 4) lemmas of `pc_sentence_words.csv` with the strongest evidence in that direction: bootstrap CI excluding 0, ranked by the association (SD difference) divided by its bootstrap standard error, so frequent and strongly associated words come before rare ones.

## Stimulus sensitivity and what the PC differences say
`pc_stimulus_sensitivity.py` (~5 s): `pc_stimulus_sensitivity.csv`, `pc_stimulus_dropped.csv` (document ids only). Control waited in a pre-surgery room with no content, VR Only watched nature videos in VR, VR Art experienced an immersive art installation, so only the VR groups can describe a stimulus. This check removes every sentence containing a stimulus word (nature-video and art-installation vocabulary, lexicon in the script; written after a first exploratory look, so not pre-specified), keeps the PCs of the full set, recomputes participant centroids from the remaining sentences and repeats the permutation ANOVA (10000 permutations, BH over 10 PCs, Holm for pairs). 35 of 324 sentences are removed (VR Art 19 of 114, VR Only 14 of 100, Control 2 of 110); all 51 participants keep sentences.

| | all sentences | without stimulus-word sentences |
|---|---|---|
| PC4 means (VR Art / VR Only / Control, SD units) | +0.33 / -0.01 / -0.36 | +0.27 / -0.15 / -0.35 |
| PC4 omnibus (FDR) | 0.003, eta^2 = 0.30 | 0.006, eta^2 = 0.28 |
| PC4 pairs (Holm): Art vs Control / Art vs Only / Only vs Control | 0.001 / 0.048 / 0.040 | 0.0006 / 0.021 / 0.16 |
| PC6 omnibus (FDR) | 0.044, eta^2 = 0.18 | 0.62, eta^2 = 0.07 |

What the PC differences say (exploratory; sentences are the Portuguese sentences of all three post-interview questions):
* **PC4 is the finding.** It is a content axis, "Experience & senses" (high) vs "Procedure & care" (low), read from typical sentences and words (below). VR Art lies high and Control low, VR Only in between. It is not a question-composition artefact (it also holds within Q1 alone, where both VR groups lie above Control, and within Q3) and survives removing the stimulus-word sentences, so VR Art differs from Control and from VR Only beyond mentions of the stimulus. The VR Only vs Control difference does not survive that removal (Holm 0.16): for VR Only it is largely carried by stimulus (nature-video) wording. It describes what participants talk about (experience and senses vs the surgical and care situation), not an emotion rating; the earlier "Enjoyment" label was only weakly supported by the words and has been replaced.
* **PC6 is not robust.** Its omnibus effect (FDR 0.044, VR Art vs Control Holm 0.017) disappears when the stimulus-word sentences are removed (FDR 0.62; the VR Art mean falls from +0.19 to -0.05), it is not significant within any single question (Q2 alone p = 0.049 with 6 to 10 participants per group), and its high end partly consists of stimulus words (*imagens*). It reads as calm/relaxed wording vs anxiety/fear wording, but it should be reported as tentative or left out.
* The sensitivity check is exploratory and covers stimulus words only; the within-question checks were run once as a scratch analysis and are not stored.

## Reading the PCs from sentences alone (no topics)
`pc_sentence_interpretation.py` (~5 s): `pc_sentence_representatives.csv`, `pc_sentence_words.csv`, `pc_sentence_word_positions.csv` (mean raw coordinates of the sentences containing each word on the 10 PCs), `pc_sentence_nuisance.csv`. Exploratory; settings fixed before running. (A) The 8 sentences most typical of each end of PC4 and PC6 (top / bottom quartile of the sentence scores, ranked by cosine similarity to the end's mean embedding, one sentence per participant). (B) For every word used by at least 5 participants (23 lemmas: spaCy `pt_core_news_sm` 3.8.0 part-of-speech tagging, keeping nouns and adjectives and lemmatising them; lemmas the tagger labels as auxiliaries anywhere in the corpus are dropped everywhere; verbs are kept only if they express a state of the speaker (`STATE_VERBS` in the script: feel, like, relax, become, seem, worry, ...), so action, communication, modal and existence verbs such as *fazer, haver, poder, passar, falar, dizer, achar, ver, tratar, saber, pensar* are dropped; vague nouns, hedges and discourse words that the tagger labels as nouns or adjectives (*coisa, vez, facto, bocadinho, pronto, mesmo*, `VAGUE_WORDS`) are dropped as well. The state-verb class and the vague-word list are explicit lists because spaCy cannot separate state from action verbs or content from placeholder words; both were written after the first tables, so they are post hoc. The first version of the table, with a plain stop-word list, gave large weights to *estou*, *seja*, *nada*, *sempre*, which is why the POS filter was introduced. Tagger limits: some first-person preterites stay unlemmatized, *bem* is tagged as an adverb and dropped, and *sinto* (a first-person form it does not lemmatize) appears next to *sentir*), the difference in mean PC score (SD units) between sentences containing it and sentences that do not, with a participant-cluster bootstrap 95% CI (2000 resamples) and BH-FDR over the words. (C) Nuisance checks: sentence length, question, and the condition difference within each question (participant-level permutation ANOVA, 10000 permutations, BH over the 6 tests).

* **PC4, high end:** *gostar, água, imagem, sítio, parecer* (lemmas; CI excludes 0; 8 of the 23 are significant at FDR 0.05 on PC4); typical sentences are short evaluative remarks about the experience ("maravilhoso", "espetacular", "emocionante"; 5 of 8 from VR Art). **Low end:** *cirurgia, cuidado, confiante, preocupação, ansiedade, ansioso, importante, pessoa, tempo*; typical sentences are state reports about the procedure and care, including calm and confident ones ("estou muito mais tranquila agora para o procedimento", "bem tratada"; 5 of 8 from Control, 7 of 8 from Q1). So PC4 contrasts evaluating an experience with reporting one's state about the procedure and care; the low end is not simply worry, so it is labelled "Procedure & care" (earlier "Surgery & care concerns").
* **PC6** (tentative): *calmo, relaxar, bom, imagem, sítio, cirurgia* at the high end ("mais calma / relaxada" after the intervention, plus the video and the surgery), *ansioso, ansiedade, preocupação, medo, importante, cuidado, pessoa* at the low end; 10 of the 23 lemmas are significant at FDR 0.05. The low end is anxiety and fear *wording*: embeddings are poor at negation, so it also contains negated mentions ("sem medo", "não diria ansiosa"), and a group difference on it does not necessarily mean more anxiety. Labelled "Calm, relaxed" (high) vs "Anxiety, fear (wording)" (low); earlier versions called it mixed.
* **Nuisance:** longer sentences score slightly lower on PC4 (Spearman rho = -0.13, 95% CI -0.27 to 0.00; PC6 0.02); question explains little (eta^2 0.04 for PC4, 0.03 for PC6). The PC4 condition difference holds within Q1 (p < 0.001, BH 0.002; both VR groups above Control) and within Q3 (p = 0.023, BH 0.069); it is not testable in Q2 (6 to 10 participants per group, p = 0.29). No PC6 test is significant within a question (BH >= 0.11).
* Limits: the vocabulary is small (23 content lemmas used by at least 5 participants), words are not stemmed, and representative sentences are short and sometimes fragments; read them as illustration, with the word table as the quantitative part.

Map axes: `semantic_space_by_condition` (no topic labels; the ends of each PC axis carry the sentence-based labels with their representative words, and the same words are drawn in the plot at the mean position of the sentences that contain them) now uses the two PCs that separate the conditions most (largest eta^2 among those passing the FDR test: PC4 and PC6), picked for display after seeing the data; the tests use the full embedding and do not depend on the axes.
