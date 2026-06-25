# FBI Recalibration in ALS — Reproducibility Package

This package contains all code and (de-identified) data needed to reproduce the
results in:

> Callegaro S et al. *Recalibration of the Frontal Behavioural Inventory cut-off for
> behavioural screening in amyotrophic lateral sclerosis.* Submitted.

---

## Package contents

```
repro_package/
├── README.md                          this file
├── requirements.txt                   exact Python dependencies
├── scripts/
│   ├── 01_build_dataset.py            from full SAV → analytic dataset
│   ├── 02_main_recalibration.py       ROC, Youden, bootstrap, LCA, κ, CV
│   ├── 03_subscales.py                Apathy + Disinhibition recalibration
│   ├── 04_sensitivity_bulbar_spinal.py    onset-stratified analysis
│   ├── 05_figures.py                  Figure 1 (ROC) + Figure 2 (calibration belt)
│   ├── 06_sensitivity_motor_severity.py   sensitivity analysis by motor severity
│   ├── 07_correlations_alsfrsr_domains.py Spearman correlations FBI vs ALSFRS-R domains
│   ├── 08_grey_zone_distribution.py   FBI score distribution across decision bands
│   ├── 09_decision_curve_analysis.py  Decision Curve Analysis for FBI cut-offs
│   └── helpers.py                     shared utility functions
├── outputs/
│   ├── log_main.txt                   full text output from every script
│   ├── log_dca.txt                    log from decision curve analysis
│   ├── table1_demographics.csv        Table 1 of the manuscript
│   ├── table2_cutoffs.csv             Table 2 of the manuscript
│   ├── table3_subscales.csv           Table 3 of the manuscript
│   ├── table4_kappa.csv               Table 4 of the manuscript
│   ├── table7_correlations.csv        Spearman correlations (script 07)
│   ├── table8_grey_zone.csv           Grey-zone distribution (script 08)
│   ├── table9_dca_nb.csv              Net benefit table (script 09)
│   └── word_tables/                   Word (.docx) versions of supplementary tables
│       ├── Table7_correlations.docx
│       ├── Table8_grey_zone.docx
│       └── Table9_dca_nb.docx
└── figures/
    ├── Figure1_ROC.png/.pdf
    ├── Figure2_calibration_belt.png/.pdf
    └── FigureS1_DCA.png/.pdf
```

---

## How to reproduce

### Requirements
- Python 3.10–3.12
- ≈ 2 GB RAM
- All dependencies listed in `requirements.txt`

### Installation
```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run all analyses
Either run the full pipeline:
```bash
cd scripts
python 01_build_dataset.py               # → data/analytic_dataset.csv  (skip if using provided file)
python 02_main_recalibration.py          # → outputs/table2_cutoffs.csv, table4_kappa.csv
python 03_subscales.py                   # → outputs/table3_subscales.csv
python 04_sensitivity_bulbar_spinal.py
python 05_figures.py                     # → figures/Figure1_*, Figure2_*
python 06_sensitivity_motor_severity.py  # → sensitivity analysis by motor severity
python 07_correlations_alsfrsr_domains.py # → outputs/table7_correlations.csv
python 08_grey_zone_distribution.py      # → outputs/table8_grey_zone.csv
python 09_decision_curve_analysis.py     # → outputs/table9_dca_nb.csv, figures/FigureS1_*
```

Or just run script `02_main_recalibration.py` to verify the headline numbers
(AUC 0.855, cut-off ≥ 9, etc.).

### Expected runtime
- `02_main_recalibration.py`: ~30 seconds (mostly bootstrap)
- `03_subscales.py`: ~10 seconds
- `04_sensitivity_bulbar_spinal.py`: ~10 seconds
- `05_figures.py`: ~15 seconds
- `06_sensitivity_motor_severity.py`: ~10 seconds
- `07_correlations_alsfrsr_domains.py`: ~15 seconds (bootstrap)
- `08_grey_zone_distribution.py`: ~5 seconds
- `09_decision_curve_analysis.py`: ~10 seconds
- **Total:** under 2 minutes on a modern laptop

### Reproducibility
All analyses use `numpy.random.seed(2026)` and `random_state=2026` in
scikit-learn / StepMix calls. Re-running on the same machine and library
versions reproduces every number in the manuscript exactly.


---

## Script descriptions

### Core pipeline (01–05)

**`01_build_dataset.py`** — Reads the source SPSS file and produces the
de-identified `analytic_dataset.csv` used by all downstream scripts. Skip
this step if using the provided CSV directly.

**`02_main_recalibration.py`** — Main analysis: ROC curve with bootstrap
95% CI for the AUC, Youden-index cut-off identification (bootstrap CI),
Latent Class Analysis (LCA-3 and LCA-7), Cohen's κ, stratified train–test
split, and 5-fold cross-validation.

**`03_subscales.py`** — Recalibration of the FBI Apathy and Disinhibition
subscales against construct-matched anchors (FrSBe-Apathy, FrSBe-Disinhibition,
ECAS-Disinhibition) and against the consensus reference standard.

**`04_sensitivity_bulbar_spinal.py`** — Onset-stratified sensitivity analysis:
ROC and Youden cut-off estimated separately for bulbar-onset (n = 76) and
spinal-onset (n = 197) patients.

**`05_figures.py`** — Produces Figure 1 (ROC curve with bootstrap CI band)
and Figure 2 (calibration belt) as PNG and PDF.

### Supplementary analyses (06–09)

**`06_sensitivity_motor_severity.py`** — Sensitivity analysis stratified by
motor severity. Tests whether the recalibrated cut-off is stable across
levels of motor disability (MiToS stage, King's stage, or ALSFRS-R tertiles),
addressing the concern that motor impairment could confound FBI scoring.

**`07_correlations_alsfrsr_domains.py`** — Construct-validity check: Spearman
rank correlations between FBI scores (total, Apathy subscale, Disinhibition
subscale) and ALSFRS-R domain scores (bulbar, fine motor, gross motor,
respiratory, total). Bootstrap 95% CIs (1 000 resamples) are reported. Under
construct validity the expected pattern is modest correlations (|ρ| < 0.30)
across all domains, confirming that motor disability does not drive FBI
scoring to a clinically meaningful degree. If only the CSV is available
(domain-level ALSFRS-R items not included), the script falls back gracefully
to reporting correlations with the ALSFRS-R total only.
Outputs: `outputs/table7_correlations.csv`, `outputs/word_tables/Table7_correlations.docx`.

**`08_grey_zone_distribution.py`** — Quantifies the distribution of FBI scores
across three clinically interpretable decision bands: rule-out zone (FBI < 9),
intermediate "grey zone" (FBI 9–14), and confirmatory rule-in zone (FBI ≥ 15).
For each band the script reports numerosity, cohort proportion, and the
proportion of consensus-impaired patients with Wilson 95% confidence intervals
(used in preference to Wald intervals given small band sizes). Results are
reported for both the full FBI cohort (n = 506) and the complete-case subset
(n = 345).
Outputs: `outputs/table8_grey_zone.csv`, `outputs/word_tables/Table8_grey_zone.docx`.

**`09_decision_curve_analysis.py`** — Decision Curve Analysis (Vickers &
Elkin 2006) quantifying the clinical utility of candidate FBI cut-offs across
the clinically relevant range of threshold probabilities (pt = 0.01–0.60).
Strategies compared: FBI continuous (quadratic logistic regression), FBI ≥ 9
(proposed cut-off), FBI ≥ 12 (confirmatory band), FBI ≥ 25 (legacy cut-off),
treat-all, and treat-none. The continuous model is estimated and evaluated
in-sample and its net-benefit estimates are therefore optimistic relative to
the fixed cut-off strategies; this limitation is flagged explicitly in the
output. Tabulated net benefit is reported at pt ∈ {0.10, 0.20, 0.25, 0.30,
0.40, 0.50} alongside interventions avoided per 100 patients versus treat-all
at pt = 0.25.
Outputs: `outputs/table9_dca_nb.csv`, `outputs/word_tables/Table9_dca_nb.docx`,
`figures/FigureS1_DCA.png/.pdf`.

---

## Data dictionary

### Inclusion criteria
- ALS diagnosis (El Escorial revised criteria)
- Valid FBI total score recorded at baseline
- Single observation per patient (follow-up reassessments excluded)

→ **N = 506 patients** in the FBI cohort
→ **N = 345 patients** in the complete-case subset (additionally have
   concurrent ECAS-behavioural, FrSBe and BBI assessments)

### Variables in `analytic_dataset.csv`

| Column | Description | Range / values |
|---|---|---|
| `patient_id` | De-identified study ID | string, unique |
| `age_at_test` | Age at neuropsychological assessment (years) | numeric (ETATEST) |
| `sex` | Sex | M / F |
| `education_years` | Years of formal education | numeric (Scol) |
| `onset_site` | Site of motor symptom onset | Bulbar / Spinal / NA (ESO_BUL_SPI) |
| `alsfrs_r_total` | ALSFRS-R total score at assessment | 0–48 |
| `mitos` | MiToS stage | 0 / 1 / 2 / 3 / 4 |
| `kings` | King's stage | 1 / 2 / 3 / 4 |
| `c9orf72` | C9orf72 expansion | Positive / Negative / Not tested |
| `strong_2017` | Strong 2017 ALS-FTSD classification | CN / ALSci / ALSbi / ALScbi / ALS-FTD |
| `fbi_total` | FBI total score | 0–72 |
| `fbi_apathy` | FBI Apathy/Negative subscale (items 1–12) | 0–36 |
| `fbi_disinhib` | FBI Disinhibition subscale (items 13–24) | 0–36 |
| `ecas_beh_patol` | ECAS-Behavioural pathological (yes=1) | 0 / 1 / NA |
| `ecas_disin_patol` | ECAS Disinhibition pathological | 0 / 1 / NA |
| `ecas_lossemp_patol` | ECAS Loss-of-empathy pathological | 0 / 1 / NA |
| `frsbe_total_patol` | FrSBe total pathological | 0 / 1 / NA |
| `frsbe_apathy_patol` | FrSBe Apathy pathological | 0 / 1 / NA |
| `frsbe_disinhib_patol` | FrSBe Disinhibition pathological | 0 / 1 / NA |
| `bbi_patol` | BBI pathological | 0 / 1 / NA |
| `in_complete_case` | Patient in the 345-patient complete-case subset (1/0) | 0 / 1 |
| `gold_consensus` | Consensus reference standard (≥2/3 of FBI-independent instruments positive) | 0 / 1 / NA |
| `gold_count` | Number of FBI-independent instruments positive | 0 / 1 / 2 / 3 / NA |

NA = missing. The consensus gold (`gold_consensus`) and `gold_count` are
defined only on the 345 complete-case patients.


## Statistical methods (verbatim from the manuscript)

### Reference standards
- **Consensus rule:** patient classified as behaviourally impaired if ≥ 2 of
  three FBI-independent instruments (ECAS-behavioural, FrSBe total, BBI)
  positive at their published pathological thresholds.
- **Latent Class Analysis (LCA-3):** Bernoulli LCA on the three indicators
  above; the latent impaired class is the one with higher overall probability
  of indicator endorsement.
- **Latent Class Analysis (LCA-7):** as above, but using 7 indicators:
  ECAS-Beh, FrSBe total, BBI, ECAS-Disinhibition, ECAS–Loss-of-empathy,
  FrSBe-Apathy, FrSBe-Disinhibition.

In each LCA, the number of classes is chosen by the Bayesian Information
Criterion (BIC). All LCA models are fit with `n_init=20` random
initialisations and `random_state=2026`.

### Discrimination
- ROC analysis, AUC with bootstrap 95% CI (2000 resamples).
- Youden index for optimal cut-off identification; bootstrap 95% CI for the
  cut-off itself (2000 resamples).

### Agreement
- Cohen's κ with the Landis–Koch interpretation (16).

### Internal validation
- Stratified 70/30 train–test split (random_state=2026).
- Stratified 5-fold cross-validation (random_state=2026).

### Subscale anchoring
- FBI Apathy vs FrSBe-Apathy (construct match).
- FBI Disinhibition vs FrSBe-Disinhibition (construct match).
- FBI Disinhibition vs ECAS-Disinhibition (independent anchor).
- Both subscales also tested against the consensus reference standard.

### Sensitivity analyses
- Bulbar (n=76) vs spinal (n=197) onset, using `ESO_BUL_SPI` from the
  original database; missing onset (87 patients in the full FBI cohort)
  excluded from this analysis.
- Motor severity stratification (MiToS / King's stage / ALSFRS-R tertiles):
  tests stability of the recalibrated cut-off across levels of motor disability.
- Construct validity: Spearman correlations between FBI and ALSFRS-R domain
  scores (script 07); |ρ| < 0.30 is the pre-specified criterion for absence
  of substantial method overlap (Streiner & Norman 2008; Gosselt et al. 2020).

### Clinical utility
- Decision Curve Analysis (Vickers & Elkin 2006) across pt = 0.01–0.60.
- Grey-zone quantification with Wilson 95% CIs (Wilson 1927).