"""
04_sensitivity_bulbar_spinal.py
================================
Sensitivity analysis: is the FBI cut-off stable across onset sites?

Produces:
  - outputs/table5_bulbar_spinal.csv
  - outputs/log_sensitivity.txt

"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, chi2

sys.path.insert(0, str(Path(__file__).parent))
from helpers import youden, perf_at, bootstrap_youden, bootstrap_ci, spawn_seeds

SEED = 2026; N_BOOT = 2000
np.random.seed(SEED)

DATA   = Path("../data/analytic_dataset.csv")
OUTDIR = Path("../outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
LOG    = open(OUTDIR / "log_sensitivity.txt", "w", encoding="utf-8")

def say(*a):
    msg = " ".join(str(x) for x in a); print(msg); LOG.write(msg + "\n"); LOG.flush()

say("="*80); say("SENSITIVITY ANALYSIS — BULBAR vs SPINAL ONSET"); say("="*80)

df = pd.read_csv(DATA)
cc = df[df['in_complete_case'] == 1].copy()

b = cc[cc['onset_site'] == 'Bulbar'].copy()
s = cc[cc['onset_site'] == 'Spinal'].copy()
say(f"\nComplete-case n={len(cc)}, with onset recorded: {len(b)+len(s)}")
say(f"  Bulbar n={len(b)} ({len(b)/(len(b)+len(s))*100:.1f}%), "
    f"gold prev = {b['gold_consensus'].mean():.1%}")
say(f"  Spinal n={len(s)} ({len(s)/(len(b)+len(s))*100:.1f}%), "
    f"gold prev = {s['gold_consensus'].mean():.1%}")

# Mann–Whitney — tests SCORE LOCATION, not the FBI-impairment relationship
say("\nMann–Whitney U tests (bulbar vs spinal raw scores):")
for col in ['fbi_total', 'fbi_apathy', 'fbi_disinhib']:
    u = mannwhitneyu(b[col], s[col])
    say(f"  {col:14s}: U={u.statistic:.0f}, p={u.pvalue:.4f}")
say("  NOTE: tests score location only — NOT whether the FBI-impairment "
    "relationship differs by group. The interaction test below is the more "
    "direct answer to the motor-confound question.")

# Per-group ROC
boot_seed_b, boot_seed_s = spawn_seeds(SEED, 2)
rows = []
for label, X, bseed in [('Bulbar', b, boot_seed_b), ('Spinal', s, boot_seed_s)]:
    ya = X['gold_consensus'].astype(int).values
    sc = X['fbi_total'].values
    r  = youden(ya, sc)
    cb, _ = bootstrap_youden(ya, sc, n_boot=N_BOOT, seed=bseed)
    lo, hi = bootstrap_ci(cb)
    p9  = perf_at(ya, sc, 9)
    p25 = perf_at(ya, sc, 25)
    say(f"\n{label} onset (n={len(X)}):")
    say(f"  AUC={r['auc']:.4f}, Youden cut-off=FBI≥{r['cut']:.0f} (95% CI {lo:.0f}–{hi:.0f})")
    say(f"  At FBI≥9 : sens={p9['sens']:.4f}, spec={p9['spec']:.4f}, κ={p9['kappa']:+.4f}")
    say(f"  At FBI≥25: sens={p25['sens']:.4f}, spec={p25['spec']:.4f}, κ={p25['kappa']:+.4f}")
    say(f"  FBI mean={X['fbi_total'].mean():.2f}±{X['fbi_total'].std():.2f}, "
        f"median={X['fbi_total'].median():.0f}")
    rows.append({
        'subgroup': label, 'n': len(X), 'gold_prevalence': ya.mean(),
        'auc': r['auc'], 'youden_cutoff': r['cut'],
        'cutoff_lo': lo, 'cutoff_hi': hi,
        'sens_at_9': p9['sens'], 'spec_at_9': p9['spec'], 'kappa_at_9': p9['kappa'],
        'sens_at_25': p25['sens'], 'spec_at_25': p25['spec'], 'kappa_at_25': p25['kappa'],
        'fbi_mean': X['fbi_total'].mean(), 'fbi_sd': X['fbi_total'].std(),
        'fbi_median': X['fbi_total'].median(),
    })

pd.DataFrame(rows).to_csv(OUTDIR / "table5_bulbar_spinal.csv", index=False)
say(f"\nWritten: {OUTDIR/'table5_bulbar_spinal.csv'}")

# ============ FBI × ONSET-SITE INTERACTION TEST ============
say("\n" + "="*80)
say("FBI × onset-site INTERACTION TEST (logistic regression)")
say("="*80)
say("Tests formally whether the FBI-impairment relationship differs by onset "
    "site (single unified model, one p-value) — more direct than comparing "
    "two separately-fit ROC curves.")

dfc = cc[['fbi_total', 'onset_site', 'gold_consensus']].dropna().copy()
dfc['bulbar']      = (dfc['onset_site'] == 'Bulbar').astype(int)
dfc['fbi_x_bulbar'] = dfc['fbi_total'] * dfc['bulbar']
X_int   = sm.add_constant(dfc[['fbi_total', 'bulbar', 'fbi_x_bulbar']])
X_noint = sm.add_constant(dfc[['fbi_total', 'bulbar']])
y_int   = dfc['gold_consensus'].astype(int)

# Guard: unpenalised Logit can hit a singular Hessian under (quasi-)complete
# separation. Fall back to L2-penalised + LR-based p-value, and say so.
p_int = None
penalised = False
try:
    mod_int = sm.Logit(y_int, X_int).fit(disp=0)
    say(mod_int.summary().as_text())
    p_int = float(mod_int.pvalues['fbi_x_bulbar'])
except np.linalg.LinAlgError:
    penalised = True
    say("  WARNING: unpenalised Logit did not converge (singular Hessian, "
        "quasi-complete separation). Refitting with L2 penalty (alpha=0.1); "
        "p-value is approximate — report this limitation explicitly.")
    mod_int  = sm.Logit(y_int, X_int).fit_regularized(alpha=0.1, disp=0)
    mod_noint = sm.Logit(y_int, X_noint).fit_regularized(alpha=0.1, disp=0)
    say(str(mod_int.params))
    lr = max(2 * (mod_int.llf - mod_noint.llf), 0)
    p_int = float(1 - chi2.cdf(lr, df=1))
    say(f"  LR statistic={lr:.3f}, approx p={p_int:.4f}")

say(f"\nFBI × bulbar interaction p-value = {p_int:.4f}"
    + (" [PENALISED, approximate]" if penalised else ""))
if p_int < 0.05:
    say("  WARNING: significant interaction — the FBI-impairment relationship "
        "differs by onset site. A single shared cut-off may not be fully "
        "justified. Consider site-specific thresholds or report as limitation.")
else:
    say("  Non-significant interaction — consistent with a shared FBI-impairment "
        "relationship across onset sites. (Absence of significance ≠ proof of "
        "no effect, especially in a single-centre moderate-sized sample.)")

# ============ AUTO-GENERATED INTERPRETATION ============
rb = rows[0]; rs = rows[1]
say("\nInterpretation (auto-generated from computed rows + interaction test):")
say(f"  κ at FBI≥9: {rb['kappa_at_9']:.3f} (bulbar), {rs['kappa_at_9']:.3f} (spinal)")
say(f"  sens/spec at FBI≥9: {rb['sens_at_9']:.2f}/{rb['spec_at_9']:.2f} (bulbar), "
    f"{rs['sens_at_9']:.2f}/{rs['spec_at_9']:.2f} (spinal)")
say(f"  sens at FBI≥25: {rb['sens_at_25']:.2f} (bulbar), {rs['sens_at_25']:.2f} (spinal)")
say(f"  Youden cut-offs: bulbar FBI≥{rb['youden_cutoff']:.0f} "
    f"[{rb['cutoff_lo']:.0f}–{rb['cutoff_hi']:.0f}], "
    f"spinal FBI≥{rs['youden_cutoff']:.0f} "
    f"[{rs['cutoff_lo']:.0f}–{rs['cutoff_hi']:.0f}]")
say(f"  Interaction p={p_int:.4f}, |Δκ|={abs(rb['kappa_at_9']-rs['kappa_at_9']):.3f}")
kd = abs(rb['kappa_at_9'] - rs['kappa_at_9'])
if kd < 0.10 and p_int >= 0.05:
    say("  Both checks agree: a single cut-off of 9 is reasonable across sites.")
elif kd >= 0.10 and p_int < 0.05:
    say("  Both checks flag a difference: review before claiming one cut-off "
        "generalises across sites.")
else:
    say(f"  The two checks disagree (|Δκ|={kd:.3f}, p={p_int:.4f}) — "
        f"inspect both before drawing a conclusion.")

LOG.close()
print("\nNext step: run 05_figures.py")