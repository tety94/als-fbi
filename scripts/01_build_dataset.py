"""
01_build_dataset.py
====================
Builds the analytic dataset (n=506) from the source PARALS SAV file.

Input:  ../source/cognitivita_FINAL_recoded_ETATEST_ECASFRSBEv3.sav
Output (DE-IDENTIFIED, safe to share / deposit):
        ../data/analytic_dataset.csv
        ../data/analytic_dataset.sav
        ../data/variable_dictionary.csv
Output (IDENTIFIABLE — internal use only, NEVER deposit/share):
        ../private_internal/patient_id_crosswalk.csv

Inclusion criteria:
  - SLA_NONSLA == 'SLA'
  - FBI_total_score not missing
  - Single baseline observation per patient (already enforced upstream)


NOTE FOR THE STATISTICIAN: this script reads the original SPSS file which
contains identifying information. If the original file is unavailable, the
analytic_dataset.csv and .sav files in ../data/ contain a de-identified copy
already prepared by this script. You can skip this script and start at 02.
"""
import sys
from pathlib import Path
import pyreadstat
import pandas as pd
import numpy as np

# ----- paths -----
SRC = Path("../source/cognitivita_FINAL_recoded_ETATEST_ECASFRSBEv3.sav")
OUTDIR = Path("../data")
OUTDIR.mkdir(parents=True, exist_ok=True)
# Identifiable outputs go in a clearly separate, clearly-named folder so
# they can never be accidentally bundled with the de-identified release.
PRIVATE_DIR = Path("../private_internal")
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    print(f"Source file not found at {SRC.resolve()}")
    print("Skip this script — the analytic dataset is already in ../data/.")
    sys.exit(0)

# ----- load -----
print(f"Loading {SRC}...")
df_all, meta = pyreadstat.read_sav(str(SRC))
print(f"Total rows in source: {len(df_all)}")

# ----- inclusion criteria -----
sla = df_all[df_all['SLA_NONSLA'] == 'SLA'].copy()
print(f"After SLA filter: n = {len(sla)}")
df = sla[sla['FBI_total_score'].notna()].copy()
print(f"After FBI valid: n = {len(df)}")

# ----- variable selection and renaming -----
mapping = {
    'CODICE'              : 'patient_id',
    'ETATEST'             : 'age_at_test',
    'Sex'                 : 'sex',
    'Scol'                : 'education_years',
    'ESO_BUL_SPI'         : 'onset_site',
    'ALSFRSRTOT'          : 'alsfrs_r_total',
    'MITOS'               : 'mitos',
    'KINGS'               : 'kings',
    'C9ORF72'             : 'c9orf72_raw',
    'STRONG2017'          : 'strong_2017',
    'FBI_total_score'     : 'fbi_total',
    'FBI_apathy'          : 'fbi_apathy',
    'FBI_disinib'         : 'fbi_disinhib',
    'ECAS_BEH_PATOL'      : 'ecas_beh_patol',
    'ECAS_DISIN_PATOL'    : 'ecas_disin_patol',
    'ECAS_LOS_EMP_PATOL'  : 'ecas_lossemp_patol',
    'FRSBE_TOTALE_PATOL'  : 'frsbe_total_patol',
    'FRSBE_APATIA_PATOL'  : 'frsbe_apathy_patol',
    'FRSBE_DISINIB_PATOL' : 'frsbe_disinhib_patol',
    'BBI_PATOL'           : 'bbi_patol',
}
missing = [s for s in mapping if s not in df.columns]
if missing:
    print(f"WARNING: missing source columns: {missing}")

# Select and rename
analytic = df[[s for s in mapping if s in df.columns]].rename(columns=mapping).copy()

# ----- recoding -----
# onset_site: 'B' -> 'Bulbar', 'S' -> 'Spinal', '' / NaN -> NaN
analytic['onset_site'] = (analytic['onset_site']
    .astype(str).str.strip()
    .replace({'B':'Bulbar', 'S':'Spinal', '':np.nan, 'nan':np.nan, 'None':np.nan}))

# c9orf72_raw: 'NO' -> 'Negative', 'C9ORF72' -> 'Positive', '' -> 'Not tested',
# true missing -> NaN. FIX: 'nan' was previously NOT mapped here, unlike the
# identical pattern used below for onset_site/strong_2017, so genuinely
# missing C9orf72 results were silently left as the literal string 'nan'
# instead of being treated as missing.
analytic['c9orf72'] = (analytic['c9orf72_raw']
    .astype(str).str.strip()
    .replace({'NO':'Negative', 'C9ORF72':'Positive', '':'Not tested',
              'nan':np.nan, 'None':np.nan}))
analytic = analytic.drop(columns=['c9orf72_raw'])

# strong_2017: blank / NaN -> NaN
analytic['strong_2017'] = (analytic['strong_2017']
    .astype(str).str.strip()
    .replace({'':np.nan, 'nan':np.nan, 'None':np.nan}))

# ----- de-identify -----
analytic = analytic.reset_index(drop=True)
analytic.insert(0, 'study_id',
                ['ALS_' + str(i+1).zfill(4) for i in range(len(analytic))])

# Write the identifiable crosswalk to the PRIVATE folder only, then drop
# patient_id from the dataframe that everything downstream (and any public
# release) will use. This used to be a commented-out line — patient_id was
# silently shipping inside analytic_dataset.csv/.sav.
if 'patient_id' in analytic.columns:
    crosswalk = analytic[['study_id', 'patient_id']].copy()
    out_crosswalk = PRIVATE_DIR / 'patient_id_crosswalk.csv'
    crosswalk.to_csv(out_crosswalk, index=False)
    print(f"Written (PRIVATE — do not share/deposit): {out_crosswalk}")
    analytic = analytic.drop(columns=['patient_id'])
else:
    print("NOTE: no patient_id column found in source — nothing to de-identify.")

# ----- consensus gold standard (excludes FBI to avoid direct circularity) -----
ANCH = ['ecas_beh_patol', 'frsbe_total_patol', 'bbi_patol']
analytic['in_complete_case'] = analytic[ANCH].notna().all(axis=1).astype(int)
analytic['gold_count'] = analytic[ANCH].astype('Int64').sum(axis=1)
mask = analytic['in_complete_case'] == 1
analytic['gold_consensus'] = pd.NA
analytic.loc[mask, 'gold_consensus'] = (analytic.loc[mask, 'gold_count'] >= 2).astype(int)

print(f"\nFinal analytic dataset: n = {len(analytic)}")
print(f"  Complete-case (FBI + 3 anchors): n = {analytic['in_complete_case'].sum()}")
print(f"  Gold-impaired (consensus ≥2/3 in complete-case): "
      f"n = {(analytic.loc[mask, 'gold_consensus']==1).sum()}")

# ----- QA: completeness of categorical variables (FIX — make missingness
# visible instead of only discoverable by manually summing Table 1 cells) -----
print("\nCategorical completeness check:")
for col in ['strong_2017', 'mitos', 'kings']:
    n_total = len(analytic)
    nn = analytic[col].notna().sum()
    flag = '' if nn == n_total else f"  <-- {n_total - nn} missing/unclassified"
    print(f"  {col:14s}: {nn}/{n_total} non-missing{flag}")

# ----- save (DE-IDENTIFIED) -----
out_csv = OUTDIR / 'analytic_dataset.csv'
out_sav = OUTDIR / 'analytic_dataset.sav'
out_dic = OUTDIR / 'variable_dictionary.csv'

analytic.to_csv(out_csv, index=False)
print(f"Written: {out_csv}")

try:
    pyreadstat.write_sav(analytic, str(out_sav))
    print(f"Written: {out_sav}")
except Exception as e:
    print(f"SAV writing failed: {e}")

# variable dictionary
dictionary = pd.DataFrame({
    'analytic_name': [c for c in analytic.columns],
    'source_name'  : [{v:k for k,v in mapping.items()}.get(c, '(derived)')
                      for c in analytic.columns],
    'description'  : {
        'study_id'             : 'De-identified sequential ID',
        'age_at_test'          : 'Age at neuropsychological assessment (years)',
        'sex'                  : 'Sex (M/F)',
        'education_years'      : 'Years of formal education',
        'onset_site'           : 'Site of motor symptom onset (Bulbar/Spinal/NA)',
        'alsfrs_r_total'       : 'ALSFRS-R total score at assessment (0–48)',
        'mitos'                : 'MiToS stage (0–4)',
        'kings'                : "King's stage (1–4)",
        'c9orf72'              : 'C9orf72 expansion (Positive/Negative/Not tested)',
        'strong_2017'          : 'Strong 2017 ALS-FTSD classification',
        'fbi_total'            : 'FBI total score (0–72)',
        'fbi_apathy'           : 'FBI Apathy/Negative subscale (items 1–12, 0–36)',
        'fbi_disinhib'         : 'FBI Disinhibition subscale (items 13–24, 0–36)',
        'ecas_beh_patol'       : 'ECAS Behavioural pathological flag (0/1)',
        'ecas_disin_patol'     : 'ECAS Disinhibition pathological flag (0/1)',
        'ecas_lossemp_patol'   : 'ECAS Loss-of-empathy pathological flag (0/1)',
        'frsbe_total_patol'    : 'FrSBe total pathological flag (0/1)',
        'frsbe_apathy_patol'   : 'FrSBe Apathy pathological flag (0/1)',
        'frsbe_disinhib_patol' : 'FrSBe Disinhibition pathological flag (0/1)',
        'bbi_patol'            : 'BBI pathological flag (0/1)',
        'in_complete_case'     : 'Patient in 345-patient complete-case subset (0/1)',
        'gold_count'           : 'Number of FBI-independent anchors positive (0–3)',
        'gold_consensus'       : 'Consensus reference standard (≥2/3 positive); NA outside complete-case',
    }.get,
    'analysis_use': [{
        'study_id': 'identifier',
        'age_at_test':'Table 1', 'sex':'Table 1', 'education_years':'Table 1',
        'onset_site':'Table 1 + Sensitivity', 'alsfrs_r_total':'Table 1',
        'mitos':'Table 1', 'kings':'Table 1', 'c9orf72':'Table 1', 'strong_2017':'Table 1',
        'fbi_total':'Primary outcome (test)',
        'fbi_apathy':'Subscale recalibration (Table 3)',
        'fbi_disinhib':'Subscale recalibration (Table 3)',
        'ecas_beh_patol':'Reference standard component',
        'ecas_disin_patol':'Subscale anchor (Table 3) + LCA-7',
        'ecas_lossemp_patol':'LCA-7',
        'frsbe_total_patol':'Reference standard component',
        'frsbe_apathy_patol':'Subscale anchor + LCA-7',
        'frsbe_disinhib_patol':'Subscale anchor + LCA-7',
        'bbi_patol':'Reference standard component',
        'in_complete_case':'Cohort flag',
        'gold_count':'Auxiliary',
        'gold_consensus':'PRIMARY REFERENCE STANDARD',
    }.get(c, '') for c in analytic.columns]
})
dictionary['description'] = dictionary['analytic_name'].map(dictionary['description'])
dictionary.to_csv(out_dic, index=False)
print(f"Written: {out_dic}")

print("\nDone. Next step: run 02_main_recalibration.py")
print("REMINDER: ../private_internal/ contains identifiable data — exclude it "
      "from any Zenodo deposit, repo, or shared archive (e.g. add it to .gitignore).")