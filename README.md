# Immersive VR

Analysis workspace for the Immersive VR study: Portuguese-language interviews, pre and post, in three groups (1 = VR Art, 2 = VR Only, 3 = Control). All data used by the current analyses is in `data/`. See `CLAUDE.md` for working rules.

## Data (`data/`)

| File | Content |
|---|---|
| `participant_conditions.csv` | `enrollment_number`, `participant_number`, `condition_number` (1 VR Art, 2 VR Only, 3 Control) |
| `immerse_interviewPost.csv` | Post interview, one row per `Participant` (= `enrollment_number`), three Portuguese text questions |
| `consensus_pre_q1.csv`, `consensus_post_q1.csv` | Rater-consensus descriptors of Q1 ("how do you feel now?"), `;`-separated, one row per participant and timepoint |
| `audio/` | Local recordings (untracked) |

Participant 33 has no condition (excluded); participant 48 has no pre descriptors; 49 participants have both timepoints.

## Analyses

Each folder has its own README with method, run order and outputs.

| Folder | Analysis | Data |
|---|---|---|
| `semantic_projection_pre/`, `semantic_projection_post/`, `semantic_projection_prepost/` | Qwen3-Embedding-0.6B projection of Q1 descriptors on a distress–relief axis (Portuguese anchors); group, and pre vs post, contrasts | `consensus_*_q1.csv` |
| `pre_post/` | Shared and per-timepoint BERTopic topics, composition, PageRank, JSD and topic-distance tests, pre vs post | `consensus_*_q1.csv` |
| `topic_network_pre/`, `topic_network_post/` | Topic flow graphs and centrality for the Q1 descriptors, per timepoint (clusters from `pre_post/`) | `consensus_*_q1.csv` |
| `q3/` | BERTopic on the care question (Q3) | `immerse_interviewPost.csv` |
| `all_questions/` | BERTopic on all three post questions | `immerse_interviewPost.csv` |
| `topic_network_post_allquestions/` | Topic network of the full post interview (the former `topic_network_post/`) | `all_questions/outputs` |
| `transcribe/` | WhisperX transcription and diarization | `data/audio/` |

## Environment

Run everything from the repository root. Create a project virtual environment and install each folder's requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r q3/requirements.txt -r all_questions/requirements.txt
```

The R scripts (`models.R`, `models_prepost.R`) need `lme4`, `lmerTest`, `clubSandwich`, `emmeans`, `dplyr`, `tidyr`, `ggplot2`, `jsonlite`. Check each folder's README for any further packages.

## Running

Each analysis is run from its own folder README. Typical order for the pre/post family:

```bash
python3 pre_post/fit_timepoint_topics.py --time pre
python3 pre_post/fit_timepoint_topics.py --time post
python3 pre_post/prepare_and_model.py
python3 pre_post/prepost_content_tests.py
python3 pre_post/prepost_pagerank_tests.py
python3 pre_post/check_topic_consistency.py
```

Topic analyses: `python3 q3/q3_topic_analysis.py`; `python3 all_questions/all_questions_topic_analysis.py` then `test_topic_differences.py` and `plot_topic_differences.py`.

Semantic projection (Portuguese descriptors): `python3 semantic_projection_prepost/project_prepost.py`, then `Rscript semantic_projection_prepost/models_prepost.R`. The `_pre` and `_post` folders are run the same way with `project.py` and `models.R`.

Generated outputs are tracked in git: re-running overwrites them.
