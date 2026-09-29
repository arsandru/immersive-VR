# Immersive VR

Analysis workspace for the Immersive VR project.

Current workflows:
- Q3 interview topic analysis using BERTopic and `Qwen/Qwen3-Embedding-0.6B`
- all-questions interview topic analysis using the same shared-topic pipeline
- semantic projection using the same Qwen embedding model
- WhisperX transcription/diarization utilities for audio processing

## Repository Layout

```text
immersive VR/
  data/
    participant_conditions.csv
    immerse_interviewPost.csv
    audio/                # local recordings, not tracked

  q3/
    q3_topic_analysis.py
    requirements.txt
    README.md
    outputs/              # generated topic-analysis outputs

  all_questions/
    all_questions_topic_analysis.py
    test_topic_differences.py
    plot_topic_differences.py
    requirements.txt
    README.md
    outputs/              # generated all-question topic outputs

  semantic_projection/
    semantic_projection.py
    word_level.R
    mean_level.R
    distress_outlier_fisher.R
    assemble_publication_figure.py
    requirements.txt
    ... primary CSV/PDF/SVG/report outputs
    robustness/           # sensitivity and comparison scripts

  transcribe/
    transcribe_whisperx.py
    README.md

```

## Q3 topic analysis

The Q3 analysis maps `Participant` in `data/immerse_interviewPost.csv` to
`enrollment_number` in `data/participant_conditions.csv`, embeds responses to
the care question, fits one shared BERTopic model, and summarizes topics by
condition.

Install and run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r q3/requirements.txt
python q3/q3_topic_analysis.py
```

See `q3/README.md` for the method, configuration, and output definitions.

## All-questions topic analysis

The parallel all-questions analysis stacks the three post-interview text
columns, preserves question identifiers, and fits one shared BERTopic model.
It reports overall and question-specific participant-weighted topic prevalence
and condition comparisons.

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

See `all_questions/README.md` for methodological and output details.

## Other inputs

Audio workflow inputs:
- `data/audio/`

## Environment

Use the project virtual environment:

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
```

Install Python dependencies:

```bash
python -m pip install -r semantic_projection/requirements.txt
python -m pip install vaderSentiment whisperx
```

Required R packages:
- `lme4`
- `lmerTest`
- `clubSandwich`
- `emmeans`
- `performance`
- `ggplot2`
- `dplyr`
- `tidyr`
- `jsonlite`

Install once in R:

```r
install.packages(c(
  "lme4", "lmerTest", "clubSandwich", "emmeans", "performance",
  "ggplot2", "dplyr", "tidyr", "jsonlite"
))
```

## Main Analysis Run Order

Run from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
```

1. Build the primary semantic projection datasets and plots

```bash
source .venv/bin/activate
python semantic_projection/semantic_projection.py
```

2. Run the word-level model

```bash
Rscript semantic_projection/word_level.R
```

3. Run the participant-level mean model

```bash
Rscript semantic_projection/mean_level.R
```

4. Run the extreme-word follow-up

```bash
Rscript semantic_projection/distress_outlier_fisher.R
```

5. Rebuild the assembled publication figure if needed

```bash
source .venv/bin/activate
python semantic_projection/assemble_publication_figure.py
```

## Current Primary Semantic Projection Outputs

Primary outputs are written directly into `semantic_projection/`.

Key files:
- `semantic_projection/semantic_projection_primary.csv`
- `semantic_projection/semantic_projection_primary_mean.csv`
- `semantic_projection/semantic_projection_primary.pdf`
- `semantic_projection/semantic_projection_primary_final.pdf`
- `semantic_projection/semantic_projection_publication_figure.svg`
- `semantic_projection/analysis_report_primary.txt`
- `semantic_projection/analysis_report_primary_mean.txt`
- `semantic_projection/diagnostics_report_primary.txt`
- `semantic_projection/diagnostics_report_primary_mean.txt`
- `semantic_projection/effect_sizes_primary.csv`
- `semantic_projection/distress_outlier_fisher_report_primary.txt`

## Notes

- The primary inferential embedding model is `Qwen/Qwen3-Embedding-0.6B` with an instruction intended to capture emotional meaning.
- Sensitivity and comparison scripts are kept under `semantic_projection/robustness/`.
- `data/audio/` is intentionally untracked so raw recordings can stay local.
- Transcription outputs are generated on demand and are not required to be present in the repository.
- Publication-facing interpretation is summarized in `final_report.md`.
