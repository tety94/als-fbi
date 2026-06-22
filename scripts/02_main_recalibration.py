"""
02_main_recalibration.py
=========================
Primary analyses for the FBI cut-off recalibration paper.

Produces:
  - outputs/table2_cutoffs.csv
  - outputs/table4_kappa.csv
  - outputs/log_main.txt

Reproducibility: seed = 2026.

"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from helpers import (youden, perf_at, bootstrap_youden, bootstrap_ci,
                     fit_lca, lca_assign_impaired, spawn_seeds, select_k_or_warn,
                     lca_entropy, lca_avg_posterior_probs, bivariate_residuals,
                     lca_blrt, cost_weighted_cutoff, bootstrap_632plus)

SEED        = 2026
N_BOOT      = 2000   # Youden CI
N_INIT_LCA  = 200    # LCA random starts (was 20)
RUN_BLRT    = True   # set False during development; slow (~3-9 min total)
N_BOOT_BLRT = 99     # >=499 for the final manuscript run
N_BOOT_632  = 500
np.random.seed(SEED)

DATA   = Path("../data/analytic_dataset.csv")
OUTDIR = Path("../outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
LOG    = open(OUTDIR / "log_main.txt", "w", encoding="utf-8")

def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); LOG.write(msg + "\n"); LOG.flush()

say("="*80)
say("FBI RECALIBRATION — MAIN ANALYSES")
say("="*80)

# ---------- load ----------
df = pd.read_csv(DATA)
say(f"\nLoaded {DATA}: n={len(df)} rows, {df.shape[1]} columns")
cc = df[df['in_complete_case'] == 1].copy()
say(f"Complete-case subset: n={len(cc)}")
y   = cc['gold_consensus'].astype(int).values
fbi = cc['fbi_total'].values
say(f"Gold-impaired prevalence: {y.mean():.1%} ({y.sum()}/{len(y)})")

boot_seed_consensus, boot_seed_lca3, boot_seed_lca7 = spawn_seeds(SEED, 3)

# ============ 1. PRIMARY ROC ============
say("\n" + "="*80)
say("1. FBI total vs CONSENSUS reference standard")
say("="*80)
r = youden(y, fbi)
say(f"AUC = {r['auc']:.4f}")
say(f"Youden cut-off: FBI ≥ {r['cut']:.0f}  "
    f"(sens={r['sens']:.4f}, spec={r['spec']:.4f}, J={r['J']:.4f})")

sys.stdout.reconfigure(encoding="utf-8")
cuts_boot, aucs_boot = bootstrap_youden(y, fbi, n_boot=N_BOOT, seed=boot_seed_consensus)
lo_c, hi_c = bootstrap_ci(cuts_boot)
lo_a, hi_a = bootstrap_ci(aucs_boot)
say(f"\nBootstrap (n={N_BOOT}, stratified):")
say(f"  AUC 95% CI    : [{lo_a:.4f}, {hi_a:.4f}]")
say(f"  Cut-off 95% CI: [{lo_c:.1f}, {hi_c:.1f}]")

say("\nCandidate cut-offs (Table 2):")
tbl2_rows = []
for c in [8, 9, 10, 12, 15, 25]:
    p = perf_at(y, fbi, c)
    say(f"  FBI ≥ {c:2d}: sens={p['sens']:.4f}  spec={p['spec']:.4f}  "
        f"ppv={p['ppv']:.4f}  npv={p['npv']:.4f}  κ={p['kappa']:+.4f}  "
        f"(TP={p['tp']}, FP={p['fp']}, TN={p['tn']}, FN={p['fn']})")
    tbl2_rows.append(p)
pd.DataFrame(tbl2_rows).to_csv(OUTDIR / "table2_cutoffs.csv", index=False)
say(f"Written: {OUTDIR/'table2_cutoffs.csv'}")

# ============ 2. LATENT CLASS ANALYSIS ============
say("\n" + "="*80)
say("2. LATENT CLASS ANALYSIS")
say("="*80)

ANCH3 = ['ecas_beh_patol', 'frsbe_total_patol', 'bbi_patol']
X3 = cc[ANCH3].astype(int).values
say(f"\nLCA-3 (ECAS-beh, FrSBe total, BBI), n_init={N_INIT_LCA}:")
fits3 = fit_lca(X3, k_values=(1, 2, 3), seed=SEED, n_init=N_INIT_LCA)
for k, f in fits3.items():
    say(f"  k={k}: BIC={f['bic']:.2f}, logLik={f['loglik']:.2f}, n_par={f['n_par']}")
K3 = 2
select_k_or_warn(fits3, K3, "LCA-3", say=say)
cc['LCA3'] = lca_assign_impaired(fits3[K3]['model'], X3)
say(f"  Prevalence (k={K3}): {cc['LCA3'].mean():.1%}")
if X3.shape[1] == 3 and K3 == 2:
    say(f"  NOTE: saturated model ({fits3[2]['n_par']} params vs 2^3-1=7 df) — "
        f"BIC comparison valid, but absolute GoF cannot be tested here.")
r3 = youden(cc['LCA3'].values, fbi)
c3, _ = bootstrap_youden(cc['LCA3'].values, fbi, n_boot=N_BOOT, seed=boot_seed_lca3)
say(f"  AUC={r3['auc']:.4f}, cut-off=FBI≥{r3['cut']:.0f}, "
    f"bootstrap CI [{np.percentile(c3,2.5):.0f}, {np.percentile(c3,97.5):.0f}]")

ANCH7 = ANCH3 + ['ecas_disin_patol', 'ecas_lossemp_patol',
                  'frsbe_apathy_patol', 'frsbe_disinhib_patol']
say(f"\nLCA-7 ({', '.join(ANCH7)}), n_init={N_INIT_LCA}:")
m7m = cc[ANCH7].notna().all(axis=1)
cc7 = cc[m7m].copy()
X7  = cc7[ANCH7].astype(int).values
say(f"  n with all 7 indicators: {len(cc7)}")
if len(cc7) != len(cc):
    say(f"  MISMATCH vs cc (n={len(cc)}) by {len(cc)-len(cc7)} — "
        f"report n={len(cc7)} for LCA-7 row in Table 4.")
fits7 = fit_lca(X7, k_values=(1, 2, 3, 4), seed=SEED, n_init=N_INIT_LCA)
for k, f in fits7.items():
    say(f"  k={k}: BIC={f['bic']:.2f}, logLik={f['loglik']:.2f}, n_par={f['n_par']}")
K7 = 2
select_k_or_warn(fits7, K7, "LCA-7", say=say)
cc7['LCA7'] = lca_assign_impaired(fits7[K7]['model'], X7)
say(f"  Prevalence (k={K7}): {cc7['LCA7'].mean():.1%}")
fbi7 = cc7['fbi_total'].values
r7 = youden(cc7['LCA7'].values, fbi7)
c7b, _ = bootstrap_youden(cc7['LCA7'].values, fbi7, n_boot=N_BOOT, seed=boot_seed_lca7)
say(f"  AUC={r7['auc']:.4f}, cut-off=FBI≥{r7['cut']:.0f}, "
    f"bootstrap CI [{np.percentile(c7b,2.5):.0f}, {np.percentile(c7b,97.5):.0f}]")

# ============ 2b. LCA DIAGNOSTICS ============
say("\n" + "="*80)
say("2b. LCA DIAGNOSTICS (entropy, AvePP, bivariate residuals, BLRT)")
say("="*80)
for label, fits, X_ind, k_used, anch in [
        ("LCA-3", fits3, X3, K3, ANCH3),
        ("LCA-7", fits7, X7, K7, ANCH7)]:
    model = fits[k_used]['model']
    ent   = lca_entropy(model, X_ind)
    avepp = lca_avg_posterior_probs(model, X_ind)
    say(f"\n{label} (k={k_used}):")
    say(f"  Entropy R = {ent:.3f}  (>=0.8 good, <0.6 poor)")
    for k, d in avepp.items():
        say(f"  Class {k}: n={d['n']}, AvePP={d['avepp']:.3f}  (>=0.7 acceptable)")
    bvr_list, saturated = bivariate_residuals(model, X_ind)
    if saturated:
        n_par  = (k_used - 1) + k_used * X_ind.shape[1]
        df_dat = 2**X_ind.shape[1] - 1
        say(f"  Bivariate residuals: NOT INFORMATIVE — model is saturated "
            f"({n_par} params vs {df_dat} df). Do not report 'no local "
            f"dependence detected' here; say the test is uninformative.")
    else:
        n_flag = sum(r['flag'] for r in bvr_list)
        say(f"  Bivariate residuals: {n_flag}/{len(bvr_list)} pairs flagged (BVR>3.84)")
        for r_ in sorted(bvr_list, key=lambda d: -d['BVR'])[:5]:
            fl = '  <-- FLAGGED' if r_['flag'] else ''
            say(f"    ({anch[r_['i']]}, {anch[r_['j']]}): BVR={r_['BVR']:.2f}{fl}")

if RUN_BLRT:
    say(f"\nBLRT k=1 vs k=2 (n_boot={N_BOOT_BLRT}; slow, several minutes):")
    for label, X_ind in [("LCA-3", X3), ("LCA-7", X7)]:
        res = lca_blrt(X_ind, 1, 2, n_boot=N_BOOT_BLRT, seed=SEED, verbose=False)
        verdict = ('>=2 classes supported' if res['p_value'] < 0.05
                   else 'does NOT reject 1-class null')
        say(f"  {label}: T_obs={res['T_obs']:.2f}, p={res['p_value']:.3f} — {verdict}")
else:
    say("\nBLRT skipped (RUN_BLRT=False).")

# ============ 3. COHEN'S κ — TABLE 4 ============
say("\n" + "="*80)
say("3. COHEN'S κ — new (FBI≥9) vs legacy (FBI≥25)")
say("="*80)
say(f"\n{'Reference':32s} {'n':>5s} {'sens@9':>8s} {'κ@9':>9s} {'κ@25':>9s} {'Δκ':>9s}")
say("-"*80)
tbl4_rows = []

def kappa_row(label, ref, score):
    p9  = perf_at(ref, score, 9)
    p25 = perf_at(ref, score, 25)
    dk  = p9['kappa'] - p25['kappa']
    say(f"{label:32s} {len(ref):5d} {p9['sens']:8.4f} {p9['kappa']:+9.4f} "
        f"{p25['kappa']:+9.4f} {dk:+9.4f}")
    return {'reference': label, 'n': len(ref),
            'sens_at_9': p9['sens'], 'kappa_at_9': p9['kappa'],
            'kappa_at_25': p25['kappa'], 'delta_kappa': dk}

for nm, col in [('ECAS-behavioural', 'ecas_beh_patol'),
                ('FrSBe total',      'frsbe_total_patol'),
                ('BBI',              'bbi_patol')]:
    tbl4_rows.append(kappa_row(nm, cc[col].astype(int).values, fbi))
tbl4_rows.append(kappa_row('Consensus ≥2/3', y, fbi))
tbl4_rows.append(kappa_row('LCA-3', cc['LCA3'].values, fbi))
tbl4_rows.append(kappa_row('LCA-7', cc7['LCA7'].values, fbi7))
tbl4 = pd.DataFrame(tbl4_rows)
tbl4.to_csv(OUTDIR / "table4_kappa.csv", index=False)
say(f"Written: {OUTDIR/'table4_kappa.csv'}")
if tbl4['n'].nunique() > 1:
    say(f"NOTE: Table 4 rows have different n ({sorted(tbl4['n'].unique())}) — "
        f"report each row's own n; do NOT use a single caption value.")
else:
    say(f"Table 4: all rows share n={tbl4['n'].iloc[0]}.")

# ============ 4. COST-WEIGHTED CUTOFF ============
say("\n" + "="*80)
say("4. COST-WEIGHTED CUTOFF (decision-theoretic, alternative to Youden's J)")
say("="*80)
say("Youden's J implicitly assumes equal cost for FN and FP (1:1). "
    "The table below shows how the cost-minimising cut-off shifts as FN "
    "are assumed increasingly more costly — the clinically realistic scenario "
    "for a screening tool where missing a case is worse than a false alarm:")
for ratio in [1, 2, 3, 5, 8]:
    rc = cost_weighted_cutoff(y, fbi, c_fn=ratio, c_fp=1.0)
    say(f"  FN:FP = {ratio}:1  ->  FBI ≥ {rc['cutoff']:.0f}  "
        f"(total cost = {rc['cost']:.0f})")

# ============ 5. INTERNAL VALIDATION ============
say("\n" + "="*80)
say("5. INTERNAL VALIDATION")
say("="*80)

# 70/30 split
idx = np.arange(len(cc))
tr_idx, te_idx = train_test_split(idx, test_size=0.30,
                                  random_state=SEED, stratify=y)
rt = youden(y[tr_idx], fbi[tr_idx])
pt = perf_at(y[te_idx], fbi[te_idx], rt['cut'])
say(f"\n70/30 split: training cut-off FBI≥{rt['cut']:.0f}, "
    f"test sens={pt['sens']:.4f}, spec={pt['spec']:.4f} "
    f"(n_test={len(te_idx)}, single split — noisier than CV below)")

# 5-fold CV
say(f"\n5-fold stratified cross-validation:")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_cuts, fold_sens, fold_spec = [], [], []
for fold, (trf, tef) in enumerate(kf.split(idx, y), 1):
    rf = youden(y[trf], fbi[trf])
    pf = perf_at(y[tef], fbi[tef], rf['cut'])
    fold_cuts.append(rf['cut']); fold_sens.append(pf['sens']); fold_spec.append(pf['spec'])
    say(f"  Fold {fold}: cut-off=FBI≥{rf['cut']:.0f}, "
        f"test sens={pf['sens']:.4f}, spec={pf['spec']:.4f}")
say(f"  Mean cut-off={np.mean(fold_cuts):.2f}, "
    f"mean sens={np.mean(fold_sens):.4f}, mean spec={np.mean(fold_spec):.4f}")

# .632+
say(f"\n.632+ bootstrap (optimism-corrected, n_boot={N_BOOT_632}):")
res632 = bootstrap_632plus(y, fbi, n_boot=N_BOOT_632, seed=SEED)
say(f"  Cut-off: FBI ≥ {res632['cutoff_full']:.0f}")
say(f"  Misclassification: apparent={res632['err_apparent']:.4f}, "
    f"OOB={res632['err_oob']:.4f}, .632+={res632['err_632plus']:.4f} "
    f"(R={res632['R_error']:.2f})")
say(f"  AUC: apparent={res632['auc_apparent']:.4f}, "
    f"OOB={res632['auc_oob']:.4f}, .632+={res632['auc_632plus']:.4f} "
    f"(R={res632['R_auc']:.2f})")
say(f"  NOTE: AUC optimism near zero is expected — FBI is a fixed instrument, "
    f"nothing is fit to compute it. Optimism concentrates in the cut-off's "
    f"misclassification rate (product of data-driven Youden optimisation).")

LOG.close()
print(f"\nFull log: {OUTDIR/'log_main.txt'}")
print("Next step: run 03_subscales.py")