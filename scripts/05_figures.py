"""
05_figures.py
==============
Produces the two manuscript figures:
  - figures/Figure1_ROC.png/.pdf
  - figures/Figure2_calibration_belt.png/.pdf
  - outputs/calibration_diagnostics.txt (Spiegelhalter z + slope/intercept)

Reproducibility: seed = 2026; matplotlib non-interactive backend.

"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from helpers import youden, perf_at, spiegelhalter_z, calibration_slope_intercept

SEED = 2026
N_BOOT = 1000
np.random.seed(SEED)

DATA = Path("../data/analytic_dataset.csv")
OUTDIR = Path("../outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR = Path("../figures"); FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.1, 'axes.labelweight': 'normal',
})

# ----- load -----
df = pd.read_csv(DATA)
cc = df[df['in_complete_case'] == 1].copy()
y  = cc['gold_consensus'].astype(int).values
fbi = cc['fbi_total'].values.astype(float)

# ============ FIGURE 1: ROC ============
fig, ax = plt.subplots(figsize=(6.5, 6.5))
fpr, tpr, thr = roc_curve(y, fbi)
auc = roc_auc_score(y, fbi)

# Operating points pulled from the SAME helpers used for Table 2.
r_opt = youden(y, fbi)
p_opt = perf_at(y, fbi, r_opt['cut'])
p25   = perf_at(y, fbi, 25)

ax.plot(fpr, tpr, color='#1f4e79', lw=2.6, label=f'AUC = {auc:.3f}')
ax.plot([0, 1], [0, 1], '--', color='grey', lw=1)
ax.scatter(1 - p_opt['spec'], p_opt['sens'], s=130, color='#c00000', zorder=5,
           label=f"FBI ≥ {r_opt['cut']:.0f} (proposed)\n"
                 f"Sens={p_opt['sens']:.2f}, Spec={p_opt['spec']:.2f}")
ax.scatter(1 - p25['spec'], p25['sens'], s=130, marker='s', color='#ed7d31', zorder=5,
           label=f"FBI ≥ 25 (legacy)\nSens={p25['sens']:.2f}, Spec={p25['spec']:.2f}")
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title(f'FBI total vs Consensus reference standard (n = {len(y)})')
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.25)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig(FIGDIR / 'Figure1_ROC.png', dpi=300, bbox_inches='tight')
plt.savefig(FIGDIR / 'Figure1_ROC.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIGDIR/'Figure1_ROC.png'}")
print(f"  Proposed: FBI≥{r_opt['cut']:.0f}  sens={p_opt['sens']:.4f} spec={p_opt['spec']:.4f}")
print(f"  Legacy  : FBI≥25          sens={p25['sens']:.4f} spec={p25['spec']:.4f}")

# ============ FIGURE 2: Calibration belt ============
# Ensure the grid always reaches at least 26 so the legacy cut-off of 25 is
# always visible regardless of how far the data's 99th percentile extends.
p99 = float(np.percentile(fbi, 99))
xmax_disp = max(p99, 26.0)
if p99 < 25:
    print(f"  NOTE: 99th percentile of FBI ({p99:.1f}) < 25 — extending xmax_disp "
          f"to {xmax_disp:.0f} so the legacy cut-off is visible in the figure.")

grid = np.linspace(0, xmax_disp, 300)
X_model = sm.add_constant(np.column_stack([fbi, fbi**2]))
mod = sm.Logit(y, X_model).fit(disp=0)
Xg  = sm.add_constant(np.column_stack([grid, grid**2]))
phat = mod.predict(Xg)

# ----- Formal calibration diagnostics (NEW) -----
# Evaluated at the ACTUAL observed scores, not the dense plotting grid.
phat_obs = mod.predict(X_model)
sh  = spiegelhalter_z(y, phat_obs)
cal = calibration_slope_intercept(y, phat_obs)
print("\nCalibration diagnostics (quadratic logistic, evaluated at observed FBI scores):")
print(f"  Spiegelhalter z = {sh['z']:.3f}, p = {sh['p_value']:.4f}  "
      f"({'MISCALIBRATION DETECTED' if sh['p_value'] < 0.05 else 'no evidence of miscalibration'})")
print(f"  Calibration intercept = {cal['intercept']:.3f} (ideal: 0), "
      f"slope = {cal['slope']:.3f} (ideal: 1; <1 = predictions too extreme/overfit)")
with open(OUTDIR / "calibration_diagnostics.txt", "w", encoding="utf-8") as fout:
    fout.write("Calibration diagnostics for the quadratic logistic calibration model\n")
    fout.write("logit(p) = b0 + b1*FBI + b2*FBI^2\n\n")
    fout.write(f"Spiegelhalter z = {sh['z']:.4f}, p = {sh['p_value']:.4f}\n")
    fout.write(f"Calibration intercept = {cal['intercept']:.4f} (ideal: 0)\n")
    fout.write(f"Calibration slope = {cal['slope']:.4f} (ideal: 1)\n")
print(f"Written: {OUTDIR/'calibration_diagnostics.txt'}")

# ----- Bootstrap confidence bands -----
rng = np.random.default_rng(SEED)
boot_curves = []
n_failed = 0
for _ in range(N_BOOT):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        n_failed += 1; continue
    try:
        Xb = sm.add_constant(np.column_stack([fbi[idx], fbi[idx]**2]))
        mb = sm.Logit(y[idx], Xb).fit(disp=0)
        boot_curves.append(mb.predict(Xg))
    except Exception:
        n_failed += 1; continue
boot_curves = np.array(boot_curves)
if n_failed:
    print(f"  [calibration bootstrap] {n_failed}/{N_BOOT} resamples failed "
          f"and were excluded — {len(boot_curves)} retained.")
lo80, hi80 = np.percentile(boot_curves, [10, 90], axis=0)
lo95, hi95 = np.percentile(boot_curves, [2.5, 97.5], axis=0)

# ----- Observed proportions in bins -----
bins = [0, 3, 6, 9, 12, 15, 18, 22, 30, 110]
tmp = cc[['fbi_total', 'gold_consensus']].copy()
tmp['gold_consensus'] = tmp['gold_consensus'].astype(int)
tmp['_bin'] = pd.cut(tmp['fbi_total'], bins=bins, right=False)
emp = (tmp.groupby('_bin', observed=False)
          .agg(p=('gold_consensus', 'mean'),
               n=('gold_consensus', 'size'),
               mid=('fbi_total', 'mean'))
          .dropna())

# ----- Cumulative sens/spec by cutoff -----
# Grid always covers at least up to xmax_disp (which is >= 26).
cutoffs = np.arange(0, int(np.ceil(xmax_disp)) + 1)
sens_arr, spec_arr = [], []
for c in cutoffs:
    pred = (fbi >= c).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    sens_arr.append(tp / (tp + fn) if (tp + fn) else np.nan)
    spec_arr.append(tn / (tn + fp) if (tn + fp) else np.nan)
sens_arr = np.array(sens_arr)
spec_arr = np.array(spec_arr)
# Explicit value-based lookup — robust if the grid start/step ever changes.
sens_at = dict(zip(cutoffs, sens_arr))
spec_at = dict(zip(cutoffs, spec_arr))

# ----- Plot -----
fig, (axb, axs2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                 gridspec_kw={'height_ratios': [1.45, 1]})
axb.fill_between(grid, lo95, hi95, color='#1f4e79', alpha=0.15, label='95% CI band')
axb.fill_between(grid, lo80, hi80, color='#1f4e79', alpha=0.30, label='80% CI band')
axb.plot(grid, phat, color='#1f4e79', lw=2.6, label='Calibrated P(impairment)')
axb.scatter(emp['mid'], emp['p'], s=emp['n'] * 3.2, color='#c00000', zorder=6,
            edgecolor='white', linewidth=1.2, alpha=0.9,
            label='Observed proportion (dot size ∝ n)')
for c, col, lab in [(9, '#2e7d32', 'New cut-off: FBI ≥ 9'),
                    (25, '#ed7d31', 'Legacy cut-off: FBI ≥ 25')]:
    axb.axvline(c, color=col, lw=2, ls='--')
    pc = mod.predict(np.array([[1, c, c**2]]))[0]
    axb.annotate(f'{lab}\nP={pc:.0%}', xy=(c, pc),
                 xytext=(c + 1.2, min(pc + 0.16, 0.95)),
                 fontsize=9.5, color=col, fontweight='bold')
axb.plot(fbi[y == 1], np.full((y == 1).sum(), 1.005),
         '|', color='#c00000', ms=6, alpha=0.5)
axb.plot(fbi[y == 0], np.full((y == 0).sum(), -0.005),
         '|', color='#7d9bc1', ms=6, alpha=0.5)
axb.set_ylabel('Probability of behavioural impairment\n'
               '(consensus gold ≥ 2/3, FBI-independent)')
axb.set_ylim(0, 1.02)
axb.set_title(
    f'Calibration belt of the FBI in ALS  (n = {len(y)}, prevalence {y.mean():.0%})\n'
    f'Spiegelhalter p={sh["p_value"]:.3f}, '
    f'slope={cal["slope"]:.2f}, intercept={cal["intercept"]:.2f}',
    fontsize=11.5, fontweight='bold')
axb.legend(loc='center right', fontsize=9.5, framealpha=0.95)
axb.grid(alpha=0.25)

axs2.plot(cutoffs, sens_arr, color='#c00000', lw=2.4, label='Sensitivity')
axs2.plot(cutoffs, spec_arr, color='#7d9bc1', lw=2.0, label='Specificity')
axs2.axvline(9, color='#2e7d32', lw=2, ls='--')
axs2.axvline(25, color='#ed7d31', lw=2, ls='--')
s9 = sens_at[9]; s25 = sens_at[25]
sp9 = spec_at[9]; sp25 = spec_at[25]
axs2.scatter([9, 25], [s9, s25], color='black', zorder=6, s=60)
axs2.text(26, (s9 + s25) / 2 + 0.05,
          f'FBI ≥ 9 : sens {s9:.0%}, spec {sp9:.0%}\n'
          f'FBI ≥ 25: sens {s25:.0%}, spec {sp25:.0%}\n'
          f'Sensitivity gain: +{(s9 - s25) * 100:.0f} pp\n'
          f'Specificity loss: −{(sp25 - sp9) * 100:.0f} pp',
          fontsize=10, fontweight='bold', va='center',
          bbox=dict(boxstyle='round,pad=0.4', fc='#fff4e6', ec='#ed7d31'))
axs2.text(9,  -0.13, 'FBI ≥ 9',  color='#2e7d32', ha='center', fontsize=9, fontweight='bold')
axs2.text(25, -0.13, 'FBI ≥ 25', color='#ed7d31', ha='center', fontsize=9, fontweight='bold')
axs2.set_xlabel('FBI score (cut-off)')
axs2.set_ylabel('Cumulative sensitivity / specificity')
axs2.set_ylim(0, 1.02)
axs2.set_xlim(0, xmax_disp)
axs2.legend(loc='center right', fontsize=9.5)
axs2.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIGDIR / 'Figure2_calibration_belt.png', dpi=300, bbox_inches='tight')
plt.savefig(FIGDIR / 'Figure2_calibration_belt.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIGDIR/'Figure2_calibration_belt.png'}")
print("\nAll figures generated.")