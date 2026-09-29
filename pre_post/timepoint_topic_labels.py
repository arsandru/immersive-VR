"""Single source of truth for the per-timepoint topic models of the Q1 descriptors.

Used by pre_post/ (core networks, topic distance), topic_network_pre/ and topic_network_post/, so the clusters' labels and the
parameters they were fitted with are identical everywhere. The clusters themselves are written once by
pre_post/fit_timepoint_topics.py ({pre,post}_units_with_topics.csv) and read by every consumer.
"""

# BERTopic settings used for BOTH timepoints (fit_timepoint_topics.py): Qwen3-Embedding-0.6B embeddings (emotional-valence
# instruction as in pre_post/prepare_and_model.py), UMAP(n_neighbors=10, n_components=5, min_dist=0, cosine, random_state=42),
# HDBSCAN(min_cluster_size=5, min_samples=2, euclidean, eom), CountVectorizer(1-2 grams, Portuguese stop words)
TOPIC_MODEL_PARAMS = {"min_cluster_size": 5, "min_samples": 2, "cluster_selection_method": "eom",
                      "umap": {"n_neighbors": 10, "n_components": 5, "min_dist": 0.0, "metric": "cosine", "random_state": 42}}

# Interpretations written after reading sample descriptors of every topic (topic ids are those of the timepoint's own model)
TOPIC_LABELS_PRE = {
    -1: "Unassigned", 0: "Anxious & nervous", 1: "Calm & tranquil", 2: "Optimism", 3: "Feeling well",
    4: "Expecting it to go well", 5: "Physical discomfort", 6: "Wanting it over", 7: "Calm but a bit anxious",
    8: "Relaxed", 9: "Trusting the team",
}
TOPIC_LABELS_POST = {
    -1: "Unassigned", 0: "Anxiety & fear", 1: "Feeling calmer", 2: "Surgery & coping", 3: "Video & nature",
    4: "Feeling well", 5: "Relaxed", 6: "Confident & calm", 7: "Comfort & peace", 8: "Fine / as usual",
    9: "Emotion & unease",
}
