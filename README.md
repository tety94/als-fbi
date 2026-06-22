# FBI Recalibration in ALS — Reproducibility Package

This package contains all code and (de-identified) data needed to reproduce the
results in:

> Chiò A et al. *Recalibration of the Frontal Behavioural Inventory cut-off for
> behavioural screening in amyotrophic lateral sclerosis.* Submitted.

---

## Package contents

```
repro_package/
├── README.md                          this file
├── requirements.txt                   exact Python dependencies
├── data/
│   ├── analytic_dataset.csv           N=506 ALS patients with valid FBI
│   ├── analytic_dataset.sav           same data in SPSS format
│   ├── variable_dictionary.csv        codebook for every variable
│   └── PATIENT_LIST.csv               patient IDs included (n=506, with subset flags)
├── scripts/
│   ├── 01_build_dataset.py            from full SAV → analytic dataset
│   ├── 02_main_recalibration.py       ROC, Youden, bootstrap, LCA, κ, CV
│   ├── 03_subscales.py                Apathy + Disinhibition recalibration
│   ├── 04_sensitivity_bulbar_spinal.py    onset-stratified analysis
│   ├── 05_figures.py                  Figure 1 (ROC) + Figure 2 (calibration belt)
│   └── helpers.py                     shared utility functions
├── outputs/
│   ├── log_main.txt                   full text output from every script
│   ├── table1_demographics.csv        Table 1 of the manuscript
│   ├── table2_cutoffs.csv             Table 2 of the manuscript
│   ├── table3_subscales.csv           Table 3 of the manuscript
│   └── table4_kappa.csv               Table 4 of the manuscript
└── figures/
    ├── Figure1_ROC.png/.pdf
    └── Figure2_calibration_belt.png/.pdf
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
python 01_build_dataset.py        # → data/analytic_dataset.csv  (skip if using provided file)
python 02_main_recalibration.py   # → outputs/table2_cutoffs.csv, table4_kappa.csv
python 03_subscales.py            # → outputs/table3_subscales.csv
python 04_sensitivity_bulbar_spinal.py
python 05_figures.py              # → figures/Figure1_*, Figure2_*
```

Or just run script `02_main_recalibration.py` to verify the headline numbers
(AUC 0.855, cut-off ≥ 9, etc.).

### Expected runtime
- `02_main_recalibration.py`: ~30 seconds (mostly bootstrap)
- `03_subscales.py`: ~10 seconds
- `04_sensitivity_bulbar_spinal.py`: ~10 seconds
- `05_figures.py`: ~15 seconds
- **Total:** under a minute on a modern laptop

### Reproducibility
All analyses use `numpy.random.seed(2026)` and `random_state=2026` in
scikit-learn / StepMix calls. Re-running on the same machine and library
versions reproduces every number in the manuscript exactly.


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

### Sensitivity analysis
- Bulbar (n=76) vs spinal (n=197) onset, using `ESO_BUL_SPI` from the
  original database; missing onset (87 patients in the full FBI cohort)
  excluded from this analysis.