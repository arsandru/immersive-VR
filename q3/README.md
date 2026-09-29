# Q3 topic analysis

This analysis examines responses to:

`Que aspetos do cuidado prestado foram mais importantes para si?`

It maps each interview response to its experimental condition, splits each
response into sentences, embeds each sentence with
`Qwen/Qwen3-Embedding-0.6B`, fits one shared BERTopic model, and compares the
resulting topics across conditions.

## Inputs

- `data/participant_conditions.csv`
  - participant key: `enrollment_number`
  - condition: `condition_number`
- `data/immerse_interviewPost.csv`
  - participant key: `Participant`
  - text: `Que aspetos do cuidado prestado foram mais importantes para si?`

Condition coding:

- `1`: VR Art
- `2`: VR Only
- `3`: Control

## Method

Each non-empty response is split at sentence-ending punctuation and line breaks.
Every resulting sentence retains its participant and condition identifiers.

Portuguese stopwords and frequent non-topical discourse terms are removed from
BERTopic's count-vectorizer representation. The original sentence, including
stopwords, is embedded by Qwen so grammatical and semantic context is preserved.
Sentences containing no substantive tokens after stopword removal are retained
in the sentence audit but excluded from topic modeling.

The script fits one shared topic model across conditions, which keeps topic IDs
comparable. It uses HDBSCAN's `leaf` selection by default to produce more
fine-grained topics in this small corpus.

Sentence splitting increases the number of modeled text units but does not
increase the number of independent participants. For condition comparisons,
prefer the participant-weighted prevalence output over raw sentence counts.

## Setup and run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r q3/requirements.txt
python q3/q3_topic_analysis.py
python q3/plot_topic_differences.py
```

To force CPU execution:

```bash
python q3/q3_topic_analysis.py --device cpu
```

## Outputs

Outputs are written to `q3/outputs/`:

- `q3_join_audit.csv`: response-level condition matches and exclusions
- `q3_sentence_audit.csv`: one row per sentence, including preprocessing status
- `q3_documents_with_topics.csv`: modeled sentences and assigned topics
- `q3_topics.csv`: overall BERTopic topic information
- `q3_topic_prevalence_by_condition.csv`: raw sentence-level prevalence
- `q3_participant_topic_proportions.csv`: each participant's topic proportions
- `q3_topic_prevalence_by_condition_participant_weighted.csv`: mean participant-level topic proportions by condition
- `q3_topic_terms_by_condition.csv`: condition-specific topic representations
- `q3_topic_plot_data.csv`: participant-weighted values, confidence intervals, and plot labels
- `q3_topic_differences_by_condition.pdf`: publication-ready condition comparison
- `q3_topic_differences_by_condition.svg`: editable vector version
- `q3_topic_differences_by_condition.png`: high-resolution raster version
- `q3_run_summary.json`: sample counts, settings, unmatched IDs, and package versions

Use `--save-model` to additionally save the fitted BERTopic model. Use
`--cluster-selection-method eom` to request broader, fewer topics.
