# Semantic Projection

This directory contains the primary semantic projection workflow used for the current publication-support analysis.

## Primary Model

The main embedding pipeline uses:
- `Qwen/Qwen3-Embedding-0.6B`
- an instruction prompt intended to capture the emotional meaning of each word or short phrase

The primary Python script is:
- `semantic_projection/semantic_projection.py`

It reads:
- `data/emotion_words_checked.csv`
- `data/Flow_current.csv`

It writes directly into `semantic_projection/`.

## Main Scripts

- `semantic_projection.py`: builds the word-level and participant-level projection datasets and the primary projection figures
- `word_level.R`: mixed-effects word-level model with CR2-robust inference and pairwise contrasts
- `mean_level.R`: participant-level mean sensitivity model
- `distress_outlier_fisher.R`: MAD0-based extreme-word follow-up and Fisher tests
- `assemble_publication_figure.py`: assembles publication-facing figure outputs

## Core Outputs

- `semantic_projection_primary.csv`
- `semantic_projection_primary_mean.csv`
- `semantic_projection_primary.pdf`
- `semantic_projection_primary_faceted.pdf`
- `semantic_projection_primary_final.pdf`
- `semantic_projection_primary_mean.pdf`
- `semantic_projection_publication_figure.svg`
- `analysis_report_primary.txt`
- `analysis_report_primary_mean.txt`
- `diagnostics_report_primary.txt`
- `diagnostics_report_primary_mean.txt`
- `effect_sizes_primary.csv`
- `distress_outlier_fisher_report_primary.txt`

## Environment

Install Python dependencies with:

```bash
source .venv/bin/activate
python -m pip install -r semantic_projection/requirements.txt
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

## Run Order

```bash
source .venv/bin/activate
python semantic_projection/semantic_projection.py
Rscript semantic_projection/word_level.R
Rscript semantic_projection/mean_level.R
Rscript semantic_projection/distress_outlier_fisher.R
python semantic_projection/assemble_publication_figure.py
```

## Sensitivity Scripts

Alternative embedding models and comparison scripts are kept in:
- `semantic_projection/robustness/`
