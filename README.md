# Immersive VR

Analysis workspace for the publication support pipeline in this repository.

Current primary workflow:
- semantic projection using `Qwen/Qwen3-Embedding-0.6B`
- word-level mixed-effects modeling with CR2-robust inference
- participant-level mean models as a sensitivity check
- WhisperX transcription/diarization utilities for audio processing

## Repository Layout

```text
flow_scripts/
  data/
    emotion_words_checked.csv
    Flow_current.csv
    audio/                # local recordings, not tracked

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

## Inputs

Core analysis inputs:
- `data/emotion_words_checked.csv`
- `data/Flow_current.csv`

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
