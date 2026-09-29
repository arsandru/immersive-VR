# All-questions topic analysis

This analysis applies the same sentence-level Qwen and BERTopic pipeline used
for Q3 to all three text columns in `data/immerse_interviewPost.csv`.

The responses are reshaped into one row per participant, question, and sentence.
One shared BERTopic model is fitted across the full corpus so topic IDs have the
same meaning across questions and conditions. Every modeled sentence retains
its `question_id` (`Q1`, `Q2`, or `Q3`), participant, and condition.

Original Portuguese sentences are embedded with
`Qwen/Qwen3-Embedding-0.6B`. Portuguese stopwords are removed only from the
BERTopic lexical representation, not from the embedding input.

The sentence-level model is configured for broad, global themes rather than
fine-grained lexical clusters. It uses HDBSCAN `eom` selection with a minimum
cluster size of 10 and `min_samples = 2`. The current solution contains 12
substantive topics and assigns 1.5% of sentences to the outlier topic.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r all_questions/requirements.txt
python all_questions/all_questions_topic_analysis.py
python all_questions/test_topic_differences.py
python all_questions/plot_topic_differences.py

# Parallel combined-response analysis
python all_questions/combined_response_topic_analysis.py
python all_questions/test_response_topic_differences.py
python all_questions/plot_response_topic_differences.py
```

Use `--device cpu` if automatic device selection is unsuitable.

## Combined-response analysis

The parallel response-level model treats each complete answer to Q1, Q2, or Q3
as one document and pools all three questions into one corpus. Raw response
prevalence is descriptive. Inference uses each participant's proportions across
their available responses, followed by per-topic Kruskal-Wallis tests with Benjamini-Hochberg FDR
correction. Pairwise Mann-Whitney tests are run only for topics passing the
FDR-corrected omnibus stage and are FDR-corrected across the three condition
contrasts. Response-topic labels are manually derived from representative
documents and stored in `combined_responses_topic_labels.csv`.

## Inference

Condition comparisons use participant-level topic proportions. Plot error bars show one standard deviation above and below each condition mean; they are descriptive and are not confidence intervals. An omnibus
Kruskal-Wallis test is calculated for each non-outlier topic. Benjamini-Hochberg adjusted p-values control the false discovery rate and are
used for primary significance decisions. Holm-adjusted values are retained in
the output tables as a conservative sensitivity analysis.

The overall tests aggregate each participant's sentences across all available
questions. The explicitly named combined-questions output instead reports raw
sentence-level prevalence after pooling Q1-Q3 into one corpus; it is descriptive
because sentences from the same participant are not independent. Question-specific tests use only participants with modeled text for
that question and apply multiplicity correction separately within each question. Topics that never occur in a question are excluded from that question's plot and correction family.

## Outputs

Combined-response outputs:

- `combined_responses_with_topics.csv`: one row per complete response
- `combined_responses_topics.csv`: response-model topic information and representative documents
- `combined_responses_topic_labels.csv`: human-reviewed response-topic labels
- `combined_responses_topic_prevalence_by_condition.csv`: raw response prevalence
- `combined_responses_participant_topic_proportions.csv`: participant-level response-topic proportions
- `combined_responses_topic_condition_tests.csv`: FDR-corrected omnibus tests
- `combined_responses_topic_pairwise_tests.csv`: gated FDR-corrected pairwise tests
- `combined_responses_topic_differences_participant_weighted.pdf`: participant-weighted means and SDs
- `combined_responses_topic_differences_raw.pdf`: descriptive raw response prevalence
- matching `.svg` and `.png` plot files

Combined-corpus sentence-level outputs:

- `combined_questions_sentence_topic_assignments.csv`: one modeled row per sentence across Q1-Q3
- `combined_questions_sentence_topic_prevalence_by_condition.csv`: raw sentence-level topic prevalence by condition
- `combined_questions_sentence_topic_plot_data.csv`: plotting values for the combined corpus
- `combined_questions_sentence_topic_differences_by_condition.pdf`: combined-corpus condition plot
- matching `.svg` and `.png` plot files

Outputs are written to `all_questions/outputs/`. They include response and
sentence audits, sentence-topic assignments, overall and question-specific
participant-weighted prevalence tables, topic terms, inferential test tables,
and condition-comparison plots. Human-reviewed English cluster labels are based on the representative documents and are stored in `all_questions_topic_labels.csv` and propagated to the other topic outputs. Topic `-1` is retained in audit outputs but
excluded from default plots and inferential tests.
