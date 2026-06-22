"""
03_subscales.py
================
Subscale recalibration: FBI Apathy (items 1–12) and FBI Disinhibition
(items 13–24), each anchored against multiple reference standards.

Produces:
  - outputs/table3_subscales.csv  (Table 3 of the manuscript)
  - outputs/log_subscales.txt

"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from helpers import youden, perf_at, bootstrap_youden, bootstrap_ci, spawn_seeds

SEED = 2026
N_BOOT = 2000
np.random.seed(SEED)

DATA = Path("../data/analytic_dataset.csv")
OUTDIR = Path("../outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
LOG = open(OUTDIR / "log_subscales.txt", "w", encoding="utf-8")
def say(*a):
    msg = " ".join(str(x) for x in a); print(msg); LOG.write(msg+"\n"); LOG.flush()

say("="*80); say("SUBSCALE RECALIBRATION"); say("="*80)

df = pd.read_csv(DATA)
cc = df[df['in_complete_case']==1].copy()
y_gold = cc['gold_consensus'].astype(int).values
N_CC = len(cc)
say(f"\nComplete-case n={N_CC}, gold prevalence={y_gold.mean():.1%}")
say(f"FBI Apathy:       mean={cc['fbi_apathy'].mean():.2f} ± {cc['fbi_apathy'].std():.2f}, "
    f"median={cc['fbi_apathy'].median():.0f}")
say(f"FBI Disinhibition: mean={cc['fbi_disinhib'].mean():.2f} ± {cc['fbi_disinhib'].std():.2f}, "
    f"median={cc['fbi_disinhib'].median():.0f}")

rows = []
# 5 bootstrap calls happen below, in this order:
#   1. FBI Apathy        vs Consensus
#   2. FBI Apathy        vs FrSBe-Apathy
#   3. FBI Disinhibition  vs Consensus
#   4. FBI Disinhibition  vs FrSBe-Disinhibition
#   5. FBI Disinhibition  vs ECAS-Disinhibition
boot_seeds = spawn_seeds(SEED, 5)
_seed_iter = iter(boot_seeds)
def next_seed():
    return next(_seed_iter)

def n_check(label, n_here):
    if n_here != N_CC:
        say(f"  NOTE: {label} anchor n={n_here} differs from full complete-case "
            f"n={N_CC} — report n={n_here} for this row in Table 3, do not assume "
            f"it equals {N_CC}.")
    else:
        say(f"  n_check: {label} anchor n={n_here} matches complete-case n={N_CC}.")

def subscale_analysis(name, subscale_col, anchor_col, anchor_label):
    """Run a Youden + bootstrap analysis for one (subscale, anchor) pair."""
    m = cc[anchor_col].notna()
    sub = cc[m]
    n_check(f"{name} vs {anchor_label}", len(sub))
    y_a = sub[anchor_col].astype(int).values
    s = sub[subscale_col].values
    r = youden(y_a, s)
    cb, ab = bootstrap_youden(y_a, s, n_boot=N_BOOT, seed=next_seed())
    lo_c, hi_c = bootstrap_ci(cb); lo_a, hi_a = bootstrap_ci(ab)
    p_at_opt = perf_at(y_a, s, r['cut'])
    say(f"\n{name} vs {anchor_label} (n={len(sub)}, prev={y_a.mean():.1%}):")
    say(f"  AUC = {r['auc']:.4f}  (95% CI {lo_a:.4f}–{hi_a:.4f})")
    say(f"  Youden cut-off = {name} ≥ {r['cut']:.0f}  (95% CI {lo_c:.0f}–{hi_c:.0f})")
    say(f"  Sens = {r['sens']:.4f}, Spec = {r['spec']:.4f}, κ = {p_at_opt['kappa']:+.4f}")
    rows.append({
        'subscale': name, 'anchor': anchor_label, 'n': len(sub),
        'prevalence': y_a.mean(), 'auc': r['auc'],
        'auc_lo': lo_a, 'auc_hi': hi_a,
        'optimal_cutoff': r['cut'],
        'cutoff_lo': lo_c, 'cutoff_hi': hi_c,
        'sensitivity': r['sens'], 'specificity': r['spec'],
        'kappa': p_at_opt['kappa'],
    })

# ----- Apathy -----
say("\n" + "-"*80); say("FBI APATHY (items 1–12, range 0–36)"); say("-"*80)
n_check("FBI Apathy vs Consensus", N_CC)
r = youden(y_gold, cc['fbi_apathy'].values)
cb, ab = bootstrap_youden(y_gold, cc['fbi_apathy'].values, n_boot=N_BOOT, seed=next_seed())
lo, hi = bootstrap_ci(cb); lo_a, hi_a = bootstrap_ci(ab)
p = perf_at(y_gold, cc['fbi_apathy'].values, r['cut'])
say(f"\nFBI Apathy vs consensus (n={N_CC}, prev={y_gold.mean():.1%}):")
say(f"  AUC = {r['auc']:.4f} (95% CI {lo_a:.4f}–{hi_a:.4f})")
say(f"  Youden cut-off = FBI-Apathy ≥ {r['cut']:.0f}  (95% CI {lo:.0f}–{hi:.0f})")
say(f"  Sens = {r['sens']:.4f}, Spec = {r['spec']:.4f}, κ = {p['kappa']:+.4f}")
rows.append({
    'subscale':'FBI Apathy','anchor':'Consensus ≥2/3','n':N_CC,
    'prevalence':y_gold.mean(),'auc':r['auc'],
    'auc_lo':lo_a,'auc_hi':hi_a,
    'optimal_cutoff':r['cut'],'cutoff_lo':lo,'cutoff_hi':hi,
    'sensitivity':r['sens'],'specificity':r['spec'],'kappa':p['kappa'],
})
say("\nCandidate cut-offs:")
for c in [5, 6, 7, 8, 9, 10]:
    pp = perf_at(y_gold, cc['fbi_apathy'].values, c)
    say(f"  ≥{c}: sens={pp['sens']:.4f}, spec={pp['spec']:.4f}, κ={pp['kappa']:+.4f}")

subscale_analysis('FBI Apathy', 'fbi_apathy', 'frsbe_apathy_patol', 'FrSBe-Apathy')

# ----- Disinhibition -----
say("\n" + "-"*80); say("FBI DISINHIBITION (items 13–24, range 0–36)"); say("-"*80)
n_check("FBI Disinhibition vs Consensus", N_CC)
r = youden(y_gold, cc['fbi_disinhib'].values)
cb, ab = bootstrap_youden(y_gold, cc['fbi_disinhib'].values, n_boot=N_BOOT, seed=next_seed())
lo, hi = bootstrap_ci(cb); lo_a, hi_a = bootstrap_ci(ab)
p = perf_at(y_gold, cc['fbi_disinhib'].values, r['cut'])
say(f"\nFBI Disinhibition vs consensus (n={N_CC}, prev={y_gold.mean():.1%}):")
say(f"  AUC = {r['auc']:.4f} (95% CI {lo_a:.4f}–{hi_a:.4f})")
say(f"  Youden cut-off = FBI-Disinhibition ≥ {r['cut']:.0f}  (95% CI {lo:.0f}–{hi:.0f})")
say(f"  Sens = {r['sens']:.4f}, Spec = {r['spec']:.4f}, κ = {p['kappa']:+.4f}")
rows.append({
    'subscale':'FBI Disinhibition','anchor':'Consensus ≥2/3','n':N_CC,
    'prevalence':y_gold.mean(),'auc':r['auc'],
    'auc_lo':lo_a,'auc_hi':hi_a,
    'optimal_cutoff':r['cut'],'cutoff_lo':lo,'cutoff_hi':hi,
    'sensitivity':r['sens'],'specificity':r['spec'],'kappa':p['kappa'],
})
say("\nCandidate cut-offs:")
for c in [1, 2, 3, 4, 5, 6, 7]:
    pp = perf_at(y_gold, cc['fbi_disinhib'].values, c)
    say(f"  ≥{c}: sens={pp['sens']:.4f}, spec={pp['spec']:.4f}, κ={pp['kappa']:+.4f}")

subscale_analysis('FBI Disinhibition', 'fbi_disinhib', 'frsbe_disinhib_patol', 'FrSBe-Disinhibition')
subscale_analysis('FBI Disinhibition', 'fbi_disinhib', 'ecas_disin_patol', 'ECAS-Disinhibition')

out = pd.DataFrame(rows)
out.to_csv(OUTDIR / "table3_subscales.csv", index=False)
say(f"\nWritten: {OUTDIR/'table3_subscales.csv'}")
if out['n'].nunique() > 1:
    say(f"\nNOTE: Table 3 rows do not all share the same n ({sorted(out['n'].unique())}) "
        f"— report each row's own n in the manuscript.")
else:
    say(f"\nTable 3: all rows share n={out['n'].iloc[0]}.")

LOG.close()
print("\nNext step: run 04_sensitivity_bulbar_spinal.py")