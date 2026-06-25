"""
helpers.py
==========
Shared utility functions for the entire FBI recalibration pipeline.

Used by:
  Primary pipeline  : 01_build_dataset through 05_figures
  Supplementary     : 06_sensitivity_motor_severity
                      07_correlations_alsfrsr_domains
                      08_grey_zone_distribution
                      09_decision_curve_analysis

"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, cohen_kappa_score


# ── ROC / Youden ──────────────────────────────────────────────────────────────

def youden(y, score):
    """
    Return AUC, Youden-optimal cut-off and operating point.

    Uses sklearn roc_curve thresholds (= observed score values), which for
    integer FBI scores produces the same Youden-optimal integer cut as an
    explicit integer grid search.

    Returns
    -------
    dict: auc, cut, sens, spec, J
    """
    y = np.asarray(y); score = np.asarray(score)
    fpr, tpr, thr = roc_curve(y, score)
    J = tpr - fpr; i = int(J.argmax())
    return {
        'auc' : float(roc_auc_score(y, score)),
        'cut' : float(thr[i]),
        'sens': float(tpr[i]),
        'spec': float(1 - fpr[i]),
        'J'   : float(J[i]),
    }


def perf_at(y, score, cutoff):
    """Sensitivity, specificity, PPV, NPV, kappa at a given cut-off (score >= cutoff)."""
    y    = np.asarray(y)
    pred = (np.asarray(score) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        'cutoff': cutoff,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'sens': float(sens), 'spec': float(spec),
        'ppv' : float(ppv),  'npv' : float(npv),
        'kappa': float(cohen_kappa_score(y, pred)),
    }


def spawn_seeds(master_seed, n):
    """
    Derive n statistically independent integer seeds from a single master
    seed via NumPy SeedSequence spawning.

    Use whenever bootstrap_youden() is called for DIFFERENT analyses in the
    same script (e.g. consensus / LCA-3 / LCA-7 / subgroups). Re-using the
    same literal seed for every call makes those analyses share the identical
    resampling sequence, silently inflating apparent mutual agreement.
    """
    ss = np.random.SeedSequence(master_seed)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(n)]


def bootstrap_youden(y, score, n_boot=2000, seed=2026, stratified=True, verbose=True):
    """
    Bootstrap distribution of Youden-optimal cut-off and AUC.

    Parameters
    ----------
    stratified : bool, default True
        Resample positives and negatives separately so every resample
        preserves the original class prevalence. Avoids degenerate
        single-class resamples when positives are ~25% of the sample.

    Returns
    -------
    cuts : np.ndarray  -- Youden-optimal cut-off in each resample
    aucs : np.ndarray  -- AUC in each resample
    """
    rng     = np.random.default_rng(seed)
    y       = np.asarray(y); score = np.asarray(score)
    n       = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    cuts, aucs, n_skipped = [], [], 0
    for _ in range(n_boot):
        if stratified:
            bp  = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            bn  = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([bp, bn])
        else:
            idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            n_skipped += 1; continue
        fpr, tpr, thr = roc_curve(y[idx], score[idx])
        cuts.append(float(thr[(tpr - fpr).argmax()]))
        aucs.append(float(roc_auc_score(y[idx], score[idx])))
    if verbose and n_skipped:
        print(f"  [bootstrap_youden] {n_skipped}/{n_boot} resamples skipped "
              f"(single-class -- should be ~0 when stratified=True)")
    return np.array(cuts), np.array(aucs)


def bootstrap_ci(arr, alpha=0.05):
    """Two-sided percentile bootstrap CI."""
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return lo, hi


# ── Proportions ───────────────────────────────────────────────────────────────

def wilson_ci(n_pos, n_total, alpha=0.05):
    """
    Wilson (1927) score confidence interval for a proportion.

    Preferred over the normal-approximation (Wald) CI for small counts,
    as occurs in some FBI score bands.

    Parameters
    ----------
    n_pos   : int   -- number of positives (e.g. consensus-impaired)
    n_total : int   -- total observations in the band
    alpha   : float -- two-sided error rate (default 0.05 -> 95% CI)

    Returns
    -------
    (lo, hi) : tuple of float
    """
    from scipy.stats import norm
    if n_total == 0:
        return (np.nan, np.nan)
    z      = norm.ppf(1 - alpha / 2)
    p      = n_pos / n_total
    denom  = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (float(max(centre - margin, 0.0)), float(min(centre + margin, 1.0)))


# ── LCA ───────────────────────────────────────────────────────────────────────

def fit_lca(X, k_values=(1, 2, 3, 4), n_init=20, max_iter=800, seed=2026):
    """
    Fit Bernoulli Latent Class Analysis for several k. Return BIC for each,
    plus the fitted model and parameter count.

    NOTE: for k=2, p=3 the model is saturated (7 params vs 2^3-1=7 df).
    StepMix .score() returns AVERAGE per-sample log-likelihood, so
    multiplying by len(X) recovers the total log-likelihood for the BIC.
    """
    from stepmix.stepmix import StepMix
    X = np.asarray(X, dtype=int)
    fits = {}
    for k in k_values:
        m = StepMix(n_components=k, measurement='bernoulli',
                    random_state=seed, n_init=n_init, max_iter=max_iter,
                    verbose=0, progress_bar=False)
        m.fit(X)
        ll    = m.score(X) * len(X)
        n_par = (k - 1) + k * X.shape[1]
        bic   = -2 * ll + n_par * np.log(len(X))
        fits[k] = {'model': m, 'bic': float(bic),
                   'loglik': float(ll), 'n_par': n_par}
    return fits


def lca_assign_impaired(model, X):
    """Per-row impaired-class membership (0/1) for a fitted StepMix model."""
    probs        = model.get_parameters()['measurement']['pis']
    impaired_cls = int(probs.sum(axis=1).argmax())
    proba        = model.predict_proba(np.asarray(X, dtype=int))
    return (proba[:, impaired_cls] >= 0.5).astype(int)


def lca_entropy(model, X):
    """
    Relative entropy R (Ramaswamy et al. 1993).
    R near 1 = well-separated classes; R < 0.6-0.8 = poor separation.
    """
    proba = model.predict_proba(np.asarray(X, dtype=int))
    n, K  = proba.shape
    ent   = -np.sum(proba * np.log(proba + 1e-12))
    return float(1 - ent / (n * np.log(K)))


def lca_avg_posterior_probs(model, X):
    """
    Average posterior probability (AvePP) per class (Nagin 2005).
    AvePP >= 0.7 per class is the conventional acceptability threshold.
    """
    proba  = model.predict_proba(np.asarray(X, dtype=int))
    assign = proba.argmax(axis=1)
    out = {}
    for k in range(proba.shape[1]):
        mask = assign == k
        out[k] = {'n': int(mask.sum()),
                  'avepp': float(proba[mask, k].mean()) if mask.sum() else float('nan')}
    return out


def bivariate_residuals(model, X, flag_threshold=3.84):
    """
    Pairwise bivariate residuals (BVR) testing local independence.
    BVR > 3.84 flags a pair with residual association not explained by
    the latent classes.

    IMPORTANT: uninformative when the model is saturated (n_par >= 2^p - 1).
    For LCA-3 (k=2, p=3) always saturated -- do not report BVR as
    evidence of local independence for that model.
    """
    params  = model.get_parameters()
    pis     = params['measurement']['pis']
    weights = params['weights']
    X       = np.asarray(X, dtype=int)
    n, p    = X.shape; K = pis.shape[0]
    saturated = ((K - 1) + K * p) >= (2**p - 1)
    results = []
    for i in range(p):
        for j in range(i + 1, p):
            obs = np.zeros((2, 2))
            for a in (0, 1):
                for b in (0, 1):
                    obs[a, b] = np.mean((X[:, i] == a) & (X[:, j] == b))
            exp = np.zeros((2, 2))
            for k in range(K):
                pi1, pj1 = pis[k, i], pis[k, j]
                marg = {(1,1): pi1*pj1, (1,0): pi1*(1-pj1),
                        (0,1): (1-pi1)*pj1, (0,0): (1-pi1)*(1-pj1)}
                for a in (0, 1):
                    for b in (0, 1):
                        exp[a, b] += weights[k] * marg[(a, b)]
            bvr = float(n * np.sum((obs - exp)**2 / np.clip(exp, 1e-8, None)))
            results.append({'i': i, 'j': j, 'BVR': bvr, 'flag': bvr > flag_threshold})
    return results, saturated


def lca_blrt(X, k_low, k_high, n_boot=99, seed=2026,
             n_init_main=50, max_iter_main=500,
             n_init_boot=10, max_iter_boot=300, verbose=True):
    """
    Bootstrap Likelihood Ratio Test (McLachlan 1987) for k_low vs k_high.
    More reliable than BIC alone (Nylund et al. 2007).
    RUNTIME: ~3-9 min per comparison with n_boot=99.
    """
    import warnings
    from stepmix.stepmix import StepMix

    def _fit(Xd, k, sd, ni, mi):
        m = StepMix(n_components=k, measurement='bernoulli', random_state=sd,
                    n_init=ni, max_iter=mi, verbose=0, progress_bar=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xd)
        return m, float(m.score(Xd) * len(Xd))

    X   = np.asarray(X, dtype=int)
    rng = np.random.default_rng(seed)
    m_low,  ll_low  = _fit(X, k_low,  seed, n_init_main, max_iter_main)
    m_high, ll_high = _fit(X, k_high, seed, n_init_main, max_iter_main)
    T_obs   = 2 * (ll_high - ll_low)
    p_low   = m_low.get_parameters()
    wts, pis_low = p_low['weights'], p_low['measurement']['pis']
    Kp, pdim = pis_low.shape
    T_boot = []
    for _ in range(n_boot):
        cls = rng.choice(Kp, size=len(X), p=wts)
        Xb  = (rng.random((len(X), pdim)) < pis_low[cls]).astype(int)
        bsd = int(rng.integers(0, 1_000_000))
        _, ll_lb = _fit(Xb, k_low,  bsd, n_init_boot, max_iter_boot)
        _, ll_hb = _fit(Xb, k_high, bsd, n_init_boot, max_iter_boot)
        T_boot.append(2 * (ll_hb - ll_lb))
    T_boot = np.array(T_boot)
    pval   = float(np.mean(T_boot >= T_obs))
    if verbose:
        print(f"  [BLRT k={k_low} vs k={k_high}] T_obs={T_obs:.2f}, "
              f"boot mean={T_boot.mean():.2f} sd={T_boot.std():.2f}, p={pval:.3f}")
    return {'T_obs': T_obs, 'T_boot': T_boot, 'p_value': pval}


# ── Decision-theoretic threshold ─────────────────────────────────────────────

def cost_weighted_cutoff(y, score, c_fn=1.0, c_fp=1.0):
    """
    Threshold minimising total cost C = c_fn*FN + c_fp*FP.
    Youden's J implicitly assumes c_fn == c_fp (1:1).
    """
    y = np.asarray(y); score = np.asarray(score)
    best_c = best_cost = None
    for c in np.unique(score):
        pred = (score >= c).astype(int)
        fn   = int(((pred == 0) & (y == 1)).sum())
        fp   = int(((pred == 1) & (y == 0)).sum())
        cost = c_fn * fn + c_fp * fp
        if best_cost is None or cost < best_cost:
            best_c, best_cost = float(c), float(cost)
    return {'cutoff': best_c, 'cost': best_cost, 'c_fn': c_fn, 'c_fp': c_fp}


# ── Calibration ───────────────────────────────────────────────────────────────

def spiegelhalter_z(y, p):
    """
    Spiegelhalter (1986) z-test for overall calibration.
    |z| > 1.96 flags miscalibration at the 5% level.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    num  = np.sum((y - p) * (1 - 2 * p))
    den  = np.sqrt(np.sum(((1 - 2 * p)**2) * p * (1 - p)))
    z    = float(num / den)
    from scipy import stats as _sps
    pval = float(2 * (1 - _sps.norm.cdf(abs(z))))
    return {'z': z, 'p_value': pval}


def calibration_slope_intercept(y, p):
    """
    Cox calibration regression (y ~ logit(p)).
    Ideal: intercept = 0, slope = 1. Slope < 1 -> predictions too extreme.
    """
    import statsmodels.api as sm
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    lp  = np.log(p / (1 - p))
    mod = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    return {'intercept': float(mod.params[0]), 'slope': float(mod.params[1])}


# ── Optimism correction ───────────────────────────────────────────────────────

def bootstrap_632plus(y, score, n_boot=200, seed=2026):
    """
    .632+ optimism-corrected AUC and misclassification rate (Efron &
    Tibshirani 1997). AUC optimism near zero is expected for a fixed
    instrument like FBI. Optimism concentrates in the misclassification
    rate (product of data-driven Youden optimisation).
    """
    rng  = np.random.default_rng(seed)
    y    = np.asarray(y); score = np.asarray(score); n = len(y)
    cut_full  = youden(y, score)['cut']
    pred_full = (score >= cut_full).astype(int)
    err_bar   = float(np.mean(pred_full != y))
    auc_full  = float(roc_auc_score(y, score))
    p_hat, q_hat = y.mean(), pred_full.mean()
    gamma = p_hat * (1 - q_hat) + (1 - p_hat) * q_hat
    oob_err_num = oob_err_den = 0.0
    oob_aucs = []
    for _ in range(n_boot):
        idx    = rng.integers(0, n, n)
        in_bag = np.zeros(n, dtype=bool); in_bag[np.unique(idx)] = True
        oob    = ~in_bag
        if oob.sum() < 5 or len(np.unique(y[idx])) < 2 or len(np.unique(y[oob])) < 2:
            continue
        cut_b    = youden(y[idx], score[idx])['cut']
        pred_oob = (score[oob] >= cut_b).astype(int)
        oob_err_num += np.sum(pred_oob != y[oob]); oob_err_den += oob.sum()
        oob_aucs.append(roc_auc_score(y[oob], score[oob]))
    err_boot0     = oob_err_num / oob_err_den
    err_boot0_auc = 1 - float(np.mean(oob_aucs))

    def _combine(eb, e0, g):
        R = float(np.clip((e0 - eb) / (g - eb) if (g - eb) else 0.0, 0, 1))
        w = 0.632 / (1 - 0.368 * R)
        return (1 - w) * eb + w * e0, R

    err_632p, R_err = _combine(err_bar,     err_boot0,     gamma)
    auc_632p, R_auc = _combine(1-auc_full,  err_boot0_auc, 0.5)
    return {
        'cutoff_full' : cut_full,
        'err_apparent': err_bar,     'err_oob'   : err_boot0,
        'err_632plus' : err_632p,    'R_error'   : R_err,
        'auc_apparent': auc_full,    'auc_oob'   : 1 - err_boot0_auc,
        'auc_632plus' : 1 - auc_632p,'R_auc'     : R_auc,
    }


# ── LCA selection audit ───────────────────────────────────────────────────────

def select_k_or_warn(fits, k_used, label, say=print):
    """
    Compare BIC-optimal k against the k used downstream.
    Logs a WARNING if they disagree.
    """
    best_k = min(fits, key=lambda k: fits[k]['bic'])
    if best_k == k_used:
        say(f"  Selected by BIC: k={best_k}  (matches k={k_used} used downstream)")
    else:
        say(f"  *** WARNING [{label}]: BIC favours k={best_k}, but pipeline uses "
            f"k={k_used}. DELIBERATE override -- disclose in manuscript. ***")
    return best_k