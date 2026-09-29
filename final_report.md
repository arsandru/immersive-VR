# Final Report: Semantic Projection Analysis

## Methods

### Design and analytic objective
We quantified emotional language along a predefined semantic axis spanning **distress** to **relief** and tested whether projection scores differed by experimental condition. Analyses were conducted at two levels: (1) **word-level** projections using all available emotion words and (2) **participant-level mean** projections computed as the mean across words per participant.

### Data sources and preprocessing
Two input files were used:
- `data/emotion_words_checked.csv`
- `data/Flow_current.csv`

Records were linked by participant `ID` after coercing `ID` to character in both files. Condition labels were merged from `Flow_current.csv` using the stripped experimental condition column (`Exp_Condition`, with fallback handling for trailing whitespace in source headers).

For semantic projection, the emotion-word field was parsed as a list and cleaned by removing tokens prefixed with `not_`. Before encoding, underscore-delimited tokens were converted to space-delimited text so that multiword expressions were represented in natural language form. Rows without usable word lists were excluded from projection steps.

### Primary embedding model and axis construction
The primary inferential model used:
- `Qwen/Qwen3-Embedding-0.6B`

Inputs were embedded with the instruction:
- `Instruct: Represent the emotional valence and intensity of this word or short phrase.`

The semantic axis was defined from non-overlapping distress and relief anchor sets:
- Distress pole: `distressed`, `danger`, `apprehensive`, `anguish`, `torment`, `agitated`
- Relief pole: `relieved`, `comforted`, `security`, `soothed`, `reassured`, `at ease`

Additional normalization rules were applied before encoding for a small number of lexical items:
- `tune_out` -> `tune out`
- `bit_worried` -> `a bit worried`
- `bit_bothered` -> `a bit bothered`
- `like` / `liked` -> `liked it`

Let `u` denote the mean pairwise difference vector between the distress and relief anchor embeddings. For each word embedding `v`, projection was computed as:

\[
\text{projection} = \frac{u \cdot v}{\|u\|}
\]

The sign of the axis was reversed after construction so that **higher values indicate stronger alignment with relief/comfort** and **lower values indicate stronger alignment with distress**.

### Outcome construction
- **Word-level outcome:** one projection score per word (`projection_fear_vs_calm`)
- **Participant-level outcome:** participant mean projection (`mean_projection_fear_vs_calm`)

Condition coding followed the analysis scripts, with condition `3` used as the reference group (Control), condition `1` corresponding to **VR+Meditation**, and condition `2` corresponding to **VR Only**.

### Statistical models

#### Word-level model (primary mixed-effects specification)
A linear mixed-effects model was fit:

\[
\text{projection}_{ij} = \beta_0 + \beta_1\,\text{condition1}_i + \beta_2\,\text{condition2}_i + b_{0i} + \varepsilon_{ij}
\]

where `b0i` is a participant-level random intercept.

Inference used **CR2 cluster-robust** variance estimation and Wald tests for three planned pairwise contrasts:
- `1 vs ref` (VR+Meditation vs Control)
- `2 vs ref` (VR Only vs Control)
- `1 vs 2` (VR+Meditation vs VR Only)

A sensitivity analysis fit a participant-clustered linear model with CR2 correction to the same word-level data.

#### Participant-mean model
A linear model was fit to participant means:

\[
\text{mean projection}_{i} = \beta_0 + \beta_1\,\text{condition1}_i + \beta_2\,\text{condition2}_i + \varepsilon_i
\]

Inference used **CR2 robust** standard errors. Pairwise contrasts were tested with CR2 Wald tests, and condition means were summarized with 95% confidence intervals.

### Extreme-word follow-up
As an exploratory follow-up, semantically extreme words were identified using a zero-centered median absolute deviation rule:

\[
MAD0 = \text{median}(|\text{projection}|)
\]

Words with projections below `-MAD0` were classified as **distress-extreme**, and words with projections above `+MAD0` were classified as **relief-extreme**. Condition differences in extreme-word frequencies were evaluated with Fisher’s exact tests.

### Sensitivity checks
Two additional embedding spaces were used as robustness checks:
- `jinaai/jina-embeddings-v5-text-small`
- `glove-wiki-gigaword-300`

The Jina model provided a modern non-Qwen embedding comparison. The 300-dimensional GloVe model was included because it is the original embedding family validated by Grand et al. (2022) for semantic projection analyses.

## Results

### Sample characteristics
- **Word-level dataset:** 135 projected words from 31 participants
  - VR+Meditation: 48 words, 11 participants
  - VR Only: 57 words, 13 participants
  - Control: 30 words, 7 participants
- **Participant-mean dataset:** 31 participants
  - VR+Meditation: n = 11
  - VR Only: n = 13
  - Control: n = 7

### Word-level mixed-effects analysis (primary Qwen model)
CR2-robust fixed effects (`projection_fear_vs_calm ~ condition + (1|Participant)`, ref = Control):
- Intercept (Control): \(\beta = 0.103\), SE = 0.037, p = 0.0481
- VR+Meditation vs Control: \(\beta = 0.123\), SE = 0.050, p = 0.0359
- VR Only vs Control: \(\beta = 0.062\), SE = 0.049, p = 0.2338

Pairwise CR2 Wald tests:
- VR+Meditation vs Control: raw \(p = 0.0140\), Bonferroni-adjusted \(p = 0.0420\)
- VR Only vs Control: raw \(p = 0.2008\), Bonferroni-adjusted \(p = 0.6023\)
- VR+Meditation vs VR Only: raw \(p = 0.1876\), Bonferroni-adjusted \(p = 0.5629\)

The participant-clustered LM sensitivity check yielded the same word-level pairwise inference pattern.

### Participant-level mean model
Model-estimated participant-level mean projection scores were:
- Control: \(M = 0.138\), 95% CI \([0.055, 0.221]\)
- VR+Meditation: \(M = 0.248\), 95% CI \([0.183, 0.314]\)
- VR Only: \(M = 0.158\), 95% CI \([0.097, 0.218]\)

CR2-robust coefficients (`mean_projection_fear_vs_calm ~ condition`, ref = Control):
- Intercept (Control): \(\beta = 0.138\), SE = 0.049, p = 0.0301
- VR+Meditation vs Control: \(\beta = 0.110\), SE = 0.056, p = 0.0718
- VR Only vs Control: \(\beta = 0.020\), SE = 0.057, p = 0.7356

Pairwise CR2 Wald tests:
- VR+Meditation vs Control: raw \(p = 0.0499\), Bonferroni-adjusted \(p = 0.1497\)
- VR Only vs Control: raw \(p = 0.7298\), Bonferroni-adjusted \(p = 1.0000\)
- VR+Meditation vs VR Only: raw \(p = 0.0247\), Bonferroni-adjusted \(p = 0.0741\)

Thus, the participant-level analysis showed the same directional pattern as the word-level model, but the pairwise contrasts did not survive Bonferroni correction.

### Main empirical pattern
In the primary Qwen embedding space, **VR+Meditation** was associated with higher relief-aligned projection scores than Control at the word level, and this contrast remained significant after Bonferroni correction. The VR Only condition was descriptively intermediate and did not differ significantly from Control or VR+Meditation after correction. The participant-level model showed the same ordering of group means but weaker inferential separation.

### Embedding-model robustness
The primary VR+Meditation versus Control effect was preserved across the sensitivity analyses:
- **Primary Qwen model:** Bonferroni-adjusted \(p = 0.0420\)
- **Jina embeddings:** Bonferroni-adjusted \(p = 0.0103\)
- **GloVe 300d:** Bonferroni-adjusted \(p = 0.00547\)

Across all three embedding spaces, the main intervention-versus-control contrast remained significant after correction. In contrast, the secondary pairwise contrasts were less stable: VR Only versus Control did not survive Bonferroni correction in either Jina or GloVe, and VR+Meditation versus VR Only also remained non-significant after correction in both sensitivity models. This convergence suggests that the principal finding is robust both to modern embedding-model choice and to the classic GloVe embedding space originally validated for semantic projection.

### Extreme-word follow-up
Using the zero-centered `MAD0` threshold, the **distress-extreme** tail was most prevalent in the Control condition (`7/30`, `23.3%`), compared with VR Only (`3/57`, `5.3%`) and VR+Meditation (`2/48`, `4.2%`). The overall Fisher test for distress-extreme word frequencies was significant, \(p = 0.0157\). By contrast, the **relief-extreme** tail was most prevalent in VR+Meditation (`25/48`, `52.1%`), followed by VR Only (`20/57`, `35.1%`) and Control (`10/30`, `33.3%`), but the overall Fisher test for relief-extreme frequencies was not significant, \(p = 0.1525\).

These follow-up results are consistent with the main projection analyses: the strongest condition difference was concentrated in the negative distress tail, where Control contributed a larger share of strongly distress-aligned words.

## Figure Caption
**Figure 1. Semantic projection results in the primary Qwen embedding space.**  
**Panel A** shows the semantic projection of emotion words within each condition on the relief-distress axis. Higher values indicate stronger alignment with relief/comfort, and lower values indicate stronger alignment with distress. The shaded vertical band marks the neutral region defined by `\(\pm MAD0\)`, where `MAD0` is the median absolute deviation from zero. Words falling to the left of this shaded region are distress-extreme, whereas words falling to the right are relief-extreme. In the primary analysis, distress-extreme words were most frequent in the Control condition (`23.3%`), compared with VR Only (`5.3%`) and VR+Meditation (`4.2%`), consistent with a significant overall Fisher test for the distress tail (\(p = 0.0157\)).  
**Panel B** shows model-estimated standardized condition means from the word-level mixed model, with vertical error bars indicating 95% confidence intervals. Brackets mark Bonferroni-significant CR2 Wald contrasts. The VR+Meditation condition showed higher relief-aligned projection scores than Control, whereas VR Only was intermediate.
