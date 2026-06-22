"""
helpers.py — shared utility functions for the recalibration analyses.

"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, cohen_kappa_score

# ------------ ROC / Youden ------------
def youden(y, score):
    """Return AUC, Youden-optimal cutoff and operating point."""
    y = np.asarray(y); score = np.asarray(score)
    fpr, tpr, thr = roc_curve(y, score)
    J = tpr - fpr; i = int(J.argmax())
    return {
        'auc': float(roc_auc_score(y, score)),
        'cut': float(thr[i]),
        'sens': float(tpr[i]),
        'spec': float(1 - fpr[i]),
        'J': float(J[i]),
    }

def perf_at(y, score, cutoff):
    """Sensitivity, specificity, PPV, NPV, κ at a given cut-off (≥)."""
    y = np.asarray(y); pred = (np.asarray(score) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        'cutoff': cutoff,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'sens': float(sens), 'spec': float(spec),
        'ppv': float(ppv), 'npv': float(npv),
        'kappa': float(cohen_kappa_score(y, pred)),
    }

def spawn_seeds(master_seed, n):
    """
    Derive n statistically independent integer seeds from a single master
    seed, using NumPy's recommended SeedSequence spawning mechanism.

    Use this whenever bootstrap_youden() (or any other seeded routine) is
    called more than once in the same script for DIFFERENT analyses (e.g.
    consensus / LCA-3 / LCA-7). Re-using the same literal seed for every
    call makes those analyses share the identical sequence of resampled
    indices, which silently inflates their apparent mutual agreement.
    """
    ss = np.random.SeedSequence(master_seed)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(n)]

def bootstrap_youden(y, score, n_boot=2000, seed=2026, stratified=True, verbose=True):
    """
    Return arrays of bootstrapped Youden cut-offs and AUCs.

    stratified=True (default): resample positives and negatives separately
    so every resample preserves the original class prevalence. This avoids
    degenerate/unstable resamples when the positive class is small (here
    ~25% of the complete-case sample).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); score = np.asarray(score)
    n = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    cuts = []; aucs = []; n_skipped = 0
    for _ in range(n_boot):
        if stratified:
            bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([bp, bn])
        else:
            idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            n_skipped += 1
            continue
        fpr, tpr, thr = roc_curve(y[idx], score[idx])
        cuts.append(float(thr[(tpr - fpr).argmax()]))
        aucs.append(float(roc_auc_score(y[idx], score[idx])))
    if verbose and n_skipped:
        print(f"  [bootstrap_youden] {n_skipped}/{n_boot} resamples skipped "
              f"(single-class — should be ~0 when stratified=True)")
    return np.array(cuts), np.array(aucs)

def bootstrap_ci(arr, alpha=0.05):
    """Percentile bootstrap CI."""
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return lo, hi

# ------------ LCA ------------
def fit_lca(X, k_values=(1, 2, 3, 4), n_init=20, max_iter=800, seed=2026):
    """
    Fit Bernoulli Latent Class Analysis for several k. Return BIC for each,
    plus the fitted model and parameter count.

    BIC parameter count: (k-1) mixing weights + k*p Bernoulli probabilities,
    where p = number of binary indicators. NOTE: for k=2, p=3 this equals 7
    free parameters against 2^3-1=7 degrees of freedom in the observed
    contingency table — the model is saturated, so absolute goodness-of-fit
    of the chosen k cannot be tested with this indicator set; only the BIC
    *comparison* across k is meaningful.

    StepMix's .score() returns the AVERAGE per-sample log-likelihood
    (verified from source), so multiplying by len(X) below correctly
    recovers the total log-likelihood used in the BIC formula.
    """
    from stepmix.stepmix import StepMix
    X = np.asarray(X, dtype=int)
    fits = {}
    for k in k_values:
        m = StepMix(n_components=k, measurement='bernoulli',
                    random_state=seed, n_init=n_init, max_iter=max_iter, verbose=0)
        m.fit(X)
        ll = m.score(X) * len(X)
        n_par = (k - 1) + k * X.shape[1]   # mixing weights + Bernoulli params
        bic = -2 * ll + n_par * np.log(len(X))
        fits[k] = {'model': m, 'bic': float(bic), 'loglik': float(ll), 'n_par': n_par}
    return fits

def lca_assign_impaired(model, X):
    """Return per-row impaired-class membership (0/1) for a fitted StepMix model."""
    probs = model.get_parameters()['measurement']['pis']
    impaired_cls = int(probs.sum(axis=1).argmax())     # class with higher endorsement
    proba = model.predict_proba(np.asarray(X, dtype=int))
    return (proba[:, impaired_cls] >= 0.5).astype(int)

def lca_entropy(model, X):
    """
    Relative entropy R of the latent class solution (Ramaswamy et al. 1993):
    R = 1 - [sum_i sum_k -p_ik*log(p_ik)] / (n*log(K)).
    R close to 1 = well-separated classes; R below ~0.6-0.8 is conventionally
    read as poor classification separability (Nagin 2005 and the wider
    LCA/growth-mixture literature use ~0.8 as a common "good" threshold, but
    treat this as a rule of thumb, not a hard cutoff).
    """
    proba = model.predict_proba(np.asarray(X, dtype=int))
    n, K = proba.shape
    eps = 1e-12
    ent = -np.sum(proba * np.log(proba + eps))
    return float(1 - ent / (n * np.log(K)))

def lca_avg_posterior_probs(model, X):
    """
    Average posterior probability (AvePP) of class membership, computed
    separately for each class among the patients assigned to it (Nagin
    2005 guideline: AvePP >= 0.7 per class is conventionally considered
    acceptable classification certainty).
    """
    proba = model.predict_proba(np.asarray(X, dtype=int))
    assign = proba.argmax(axis=1)
    out = {}
    for k in range(proba.shape[1]):
        mask = assign == k
        out[k] = {'n': int(mask.sum()),
                  'avepp': float(proba[mask, k].mean()) if mask.sum() else float('nan')}
    return out

def bivariate_residuals(model, X, flag_threshold=3.84):
    """
    Pairwise bivariate residuals (BVR) testing the LOCAL INDEPENDENCE
    assumption of the LCA: for every pair of indicators (i,j), compares the
    observed 2x2 joint frequency table to the table implied by the fitted
    model under conditional independence given class. A large BVR (rule of
    thumb: > ~3.84, i.e. roughly a chi-square(1) 5% critical value) flags a
    pair whose residual association the latent classes do not explain.

    IMPORTANT CAVEAT, verified by simulation: this diagnostic has NO POWER
    when the model is saturated or has very few residual degrees of
    freedom. With p binary indicators there are 2^p - 1 effective df in the
    joint table; if n_par (see fit_lca) >= 2^p - 1, the model can match the
    data's joint distribution essentially exactly regardless of whether
    local independence truly holds, and BVR will trivially be ~0 for every
    pair. This is exactly the situation for LCA-3 (3 indicators, k=2: 7
    parameters vs 7 df) — BVR is NOT an informative diagnostic for that
    specific model and should not be reported as if it were. It IS
    informative for LCA-7 (7 indicators, k=2: 15 parameters vs 127 df).

    Also note: testing many pairs at once inflates the false-positive rate
    of the threshold heuristic (multiple comparisons) — treat flags as
    suggestive, not as formal independent significance tests.
    """
    params = model.get_parameters()
    pis = params['measurement']['pis']        # (K, p): P(X_j=1 | class k)
    weights = params['weights']                # (K,)
    X = np.asarray(X, dtype=int)
    n, p = X.shape
    K = pis.shape[0]
    n_par = (K - 1) + K * p
    df_data = 2**p - 1
    saturated = n_par >= df_data

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
                marg = {(1, 1): pi1 * pj1, (1, 0): pi1 * (1 - pj1),
                        (0, 1): (1 - pi1) * pj1, (0, 0): (1 - pi1) * (1 - pj1)}
                for a in (0, 1):
                    for b in (0, 1):
                        exp[a, b] += weights[k] * marg[(a, b)]
            bvr = float(n * np.sum((obs - exp) ** 2 / np.clip(exp, 1e-8, None)))
            results.append({'i': i, 'j': j, 'BVR': bvr, 'flag': bvr > flag_threshold})
    return results, saturated

def lca_blrt(X, k_low, k_high, n_boot=99, seed=2026,
             n_init_main=50, max_iter_main=500,
             n_init_boot=10, max_iter_boot=300, verbose=True):
    """
    Bootstrap Likelihood Ratio Test (McLachlan 1987) for k_low vs k_high
    latent classes. Preferred over the asymptotic chi-square LRT (invalid
    here due to boundary/non-regularity of the class-count testing problem)
    and, per Nylund et al. (2007)'s simulation study, generally more
    reliable than BIC alone or the Lo-Mendell-Rubin test for this purpose.

    RUNTIME WARNING: fits 2*(n_boot+1) StepMix models. With n_boot=99 this
    typically takes several minutes per (k_low, k_high) comparison — budget
    accordingly, and consider running with RUN_BLRT=False during iterative
    development (see 02_main_recalibration.py).
    """
    import warnings
    from stepmix.stepmix import StepMix

    def _fit(Xd, k, sd, n_init, max_iter):
        m = StepMix(n_components=k, measurement='bernoulli', random_state=sd,
                    n_init=n_init, max_iter=max_iter, verbose=0, progress_bar=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xd)
        ll = m.score(Xd) * len(Xd)
        return m, float(ll)

    X = np.asarray(X, dtype=int)
    rng = np.random.default_rng(seed)

    m_low, ll_low = _fit(X, k_low, seed, n_init_main, max_iter_main)
    m_high, ll_high = _fit(X, k_high, seed, n_init_main, max_iter_main)
    T_obs = 2 * (ll_high - ll_low)

    p_low = m_low.get_parameters()
    weights_low, pis_low = p_low['weights'], p_low['measurement']['pis']
    Kp, p_dim = pis_low.shape

    T_boot = []
    for b in range(n_boot):
        cls = rng.choice(Kp, size=len(X), p=weights_low)
        Xb = (rng.random((len(X), p_dim)) < pis_low[cls]).astype(int)
        bseed = int(rng.integers(0, 1_000_000))
        _, ll_low_b = _fit(Xb, k_low, bseed, n_init_boot, max_iter_boot)
        _, ll_high_b = _fit(Xb, k_high, bseed, n_init_boot, max_iter_boot)
        T_boot.append(2 * (ll_high_b - ll_low_b))
    T_boot = np.array(T_boot)
    pval = float(np.mean(T_boot >= T_obs))
    if verbose:
        print(f"  [BLRT k={k_low} vs k={k_high}] T_obs={T_obs:.2f}, "
              f"bootstrap-null mean={T_boot.mean():.2f} sd={T_boot.std():.2f}, p={pval:.3f}")
    return {'T_obs': T_obs, 'T_boot': T_boot, 'p_value': pval}

def cost_weighted_cutoff(y, score, c_fn=1.0, c_fp=1.0):
    """
    Threshold minimising total misclassification cost C = c_fn*FN + c_fp*FP,
    instead of Youden's J (which implicitly assumes c_fn == c_fp). Use with
    several illustrative cost ratios to show how the recommended cut-off
    shifts as missing a true case is assumed to be more costly than a false
    alarm — the decision-theoretic framing requested for clinical screening
    thresholds.
    """
    y = np.asarray(y); score = np.asarray(score)
    best_c, best_cost = None, None
    for c in np.unique(score):
        pred = (score >= c).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        cost = c_fn * fn + c_fp * fp
        if best_cost is None or cost < best_cost:
            best_c, best_cost = float(c), float(cost)
    return {'cutoff': best_c, 'cost': best_cost, 'c_fn': c_fn, 'c_fp': c_fp}

def spiegelhalter_z(y, p):
    """
    Spiegelhalter (1986) z-test for overall calibration of predicted
    probabilities p against observed binary outcomes y. Under H0 (perfect
    calibration), z ~ N(0,1); |z| > ~1.96 flags miscalibration.
    """
    y = np.asarray(y, dtype=float); p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum(((1 - 2 * p) ** 2) * p * (1 - p)))
    z = float(num / den)
    from scipy import stats as _sps
    pval = float(2 * (1 - _sps.norm.cdf(abs(z))))
    return {'z': z, 'p_value': pval}

def calibration_slope_intercept(y, p):
    """
    Cox calibration regression: fit y ~ logit(p) and return (intercept,
    slope). Perfect calibration -> intercept=0, slope=1. Slope < 1
    indicates predictions are too extreme (overfit / need shrinkage);
    a non-zero intercept indicates systematic over/under-prediction.
    """
    import statsmodels.api as sm
    y = np.asarray(y); p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    X = sm.add_constant(lp)
    mod = sm.Logit(y, X).fit(disp=0)
    return {'intercept': float(mod.params[0]), 'slope': float(mod.params[1])}

def bootstrap_632plus(y, score, n_boot=200, seed=2026):
    """
    .632+ optimism-corrected estimate (Efron & Tibshirani 1997) of both (a)
    the misclassification rate at the Youden-optimal cut-off and (b) AUC.
    Blends the apparent (in-sample, optimistic) estimate with an
    out-of-bag bootstrap estimate, weighted by how much overfitting is
    detected relative to a "no-information" baseline. More principled than
    reporting only 5-fold CV alongside the full-sample point estimate.

    NOTE: AUC here typically shows near-zero optimism, and that is
    expected, not a bug — AUC has no fitted/selected free parameter (the
    score is a fixed instrument, not something estimated from the sample),
    so there is little for .632+ to correct. The cut-off's misclassification
    rate, by contrast, IS the result of a data-driven selection step
    (Youden optimisation) and typically shows visible optimism.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); score = np.asarray(score); n = len(y)

    cut_full = youden(y, score)['cut']
    pred_full = (score >= cut_full).astype(int)
    err_bar = float(np.mean(pred_full != y))
    auc_full = float(roc_auc_score(y, score))
    err_bar_auc = 1 - auc_full
    p_hat, q_hat = y.mean(), pred_full.mean()
    gamma = p_hat * (1 - q_hat) + (1 - p_hat) * q_hat

    oob_err_num, oob_err_den = 0.0, 0
    oob_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        in_bag = np.zeros(n, dtype=bool); in_bag[np.unique(idx)] = True
        oob = ~in_bag
        if oob.sum() < 5 or len(np.unique(y[idx])) < 2 or len(np.unique(y[oob])) < 2:
            continue
        cut_b = youden(y[idx], score[idx])['cut']
        pred_oob = (score[oob] >= cut_b).astype(int)
        oob_err_num += np.sum(pred_oob != y[oob]); oob_err_den += oob.sum()
        oob_aucs.append(roc_auc_score(y[oob], score[oob]))

    err_boot0 = oob_err_num / oob_err_den
    err_boot0_auc = 1 - float(np.mean(oob_aucs))

    def _combine(eb, e0, g):
        R = (e0 - eb) / (g - eb) if (g - eb) != 0 else 0.0
        R = float(np.clip(R, 0, 1))
        w = 0.632 / (1 - 0.368 * R)
        return (1 - w) * eb + w * e0, R, w

    err_632p, R, w = _combine(err_bar, err_boot0, gamma)
    err_632p_auc, R_auc, w_auc = _combine(err_bar_auc, err_boot0_auc, 0.5)

    return {
        'cutoff_full': cut_full,
        'err_apparent': err_bar, 'err_oob': err_boot0, 'err_632plus': err_632p,
        'auc_apparent': auc_full, 'auc_oob': 1 - err_boot0_auc, 'auc_632plus': 1 - err_632p_auc,
        'R_error': R, 'R_auc': R_auc,
    }

def select_k_or_warn(fits, k_used, label, say=print):
    """
    Compare the BIC-optimal k against the k the pipeline is actually going
    to use downstream (k_used), and make the decision explicit instead of
    silently overriding BIC. Always logs whether the two agree; loudly
    warns if they don't, so a manuscript claim of "selected by BIC" can
    never silently become inaccurate.
    """
    best_k = min(fits, key=lambda k: fits[k]['bic'])
    if best_k == k_used:
        say(f"  Selected by BIC: k={best_k}  (matches the k={k_used} used downstream)")
    else:
        say(f"  *** WARNING [{label}]: BIC favours k={best_k}, but the pipeline is "
            f"configured to use k={k_used} for the binary impaired/unimpaired "
            f"framework. This is a DELIBERATE override for interpretability, NOT "
            f"a BIC-driven choice — disclose this explicitly in the manuscript "
            f"if this warning fires. ***")
    return best_k