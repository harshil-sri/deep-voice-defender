"""Official ASVspoof 2019 LA metrics: EER, minimum normalized t-DCF, per-attack EER, bootstrap CI.

The t-DCF implementation follows the ASVspoof 2019 evaluation plan (v0.4, Jan 2019),
Sections 5.1-5.2 (pages 8-10), reproduced here so the code is self-contained:

  t-DCF(s) = C1 * P_cm_miss(s) + C2 * P_cm_fa(s)                                   (Eq. 1)

  C1 = pi_tar * (C_miss_cm - C_miss_asv * P_asv_miss) - pi_non * C_fa_asv * P_asv_fa
  C2 = C_fa_cm * pi_spoof * (1 - P_asv_miss_spoof)

  normalized:  t-DCF_norm(s) = t-DCF(s) / min(C1, C2)                              (Eq. 2)
  min over all CM thresholds s (optimal threshold picked on the scored set, Eq. 5).

  LA constants (Table 2): pi_tar=0.9405, pi_non=0.0095, pi_spoof=0.05,
  C_miss_asv=1, C_fa_asv=10, C_miss_cm=1, C_fa_cm=10.

  The fixed ASV operates at its own EER point: P_asv_miss = P_asv_fa at the
  EER threshold computed from ASV scores of target vs non-target trials.
  P_asv_miss_spoof = proportion of spoof trials rejected by the ASV at that
  threshold.

Labels/scores convention: label 1 = bonafide (positive), score high = bonafide.
"""

import numpy as np

# Table 2 of the evaluation plan (logical access)
PI_TAR = 0.9405
PI_NON = 0.0095
PI_SPOOF = 0.05
C_MISS_ASV = 1.0
C_FA_ASV = 10.0
C_MISS_CM = 1.0
C_FA_CM = 10.0


def _error_rates_at_threshold(scores_pos, scores_neg, threshold):
    """Miss rate of positives (score <= thr) and false-alarm rate of negatives (score > thr)."""
    p_miss = np.mean(scores_pos <= threshold)
    p_fa = np.mean(scores_neg > threshold)
    return p_miss, p_fa


def compute_eer(scores_pos, scores_neg):
    """EER from pooled positive (bonafide) and negative (spoof) scores.

    Uses the interpolation-on-ROC approach: find threshold where miss == fa.
    Returns (eer, threshold_at_eer).
    """
    all_scores = np.concatenate([scores_pos, scores_neg])
    thresholds = np.unique(all_scores)
    # Candidate thresholds strictly between classes; add -inf/+inf edges.
    candidates = np.concatenate([[-np.inf], thresholds, [np.inf]])
    misses = np.array([_error_rates_at_threshold(scores_pos, scores_neg, t)[0] for t in candidates])
    fas = np.array([_error_rates_at_threshold(scores_pos, scores_neg, t)[1] for t in candidates])
    diff = misses - fas
    # Find sign change of diff; linearly interpolate between the two points.
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        eer = abs(diff[0]) / 2.0
        return float(eer), float(candidates[0])
    i = idx[0]
    d1, d2 = diff[i], diff[i + 1]
    if d2 - d1 == 0:
        eer = misses[i]
    else:
        w = d1 / (d1 - d2)
        eer = misses[i] + w * (misses[i + 1] - misses[i])
    return float(np.clip(eer, 0.0, 1.0)), float(candidates[i])


def compute_asv_operating_point(tar_asv, non_asv, spoof_asv):
    """Fixed-ASV error rates at the ASV's own EER threshold (target vs non-target)."""
    eer, thr = compute_eer(tar_asv, non_asv)
    p_miss_asv, p_fa_asv = _error_rates_at_threshold(tar_asv, non_asv, thr)
    p_miss_asv_spoof = np.mean(spoof_asv <= thr)
    return {
        "asv_eer": eer,
        "asv_threshold": thr,
        "p_miss_asv": float(p_miss_asv),
        "p_fa_asv": float(p_fa_asv),
        "p_miss_asv_spoof": float(p_miss_asv_spoof),
    }


def compute_min_tdcf(cm_scores, cm_labels, asv=None, asv_params=None):
    """Official minimum normalized t-DCF.

    Args:
        cm_scores: (N,) higher = more bonafide.
        cm_labels: (N,) 1 = bonafide, 0 = spoof.
        asv: dict from `load_asv_trials` (tar_asv/non_asv/spoof_asv). Required
            unless `asv_params` (the output of `compute_asv_operating_point`)
            is passed explicitly (e.g. to reuse a cached operating point).
    Returns:
        (min_tdcf, info_dict) — info_dict contains the ASV operating point,
        C1, C2, alpha/beta weights, and the optimal CM threshold.
    """
    cm_scores = np.asarray(cm_scores, dtype=np.float64)
    cm_labels = np.asarray(cm_labels)
    if asv_params is None:
        if asv is None:
            raise ValueError("Provide either `asv` trial scores or `asv_params`")
        asv_params = compute_asv_operating_point(asv["tar_asv"], asv["non_asv"], asv["spoof_asv"])

    c1 = PI_TAR * (C_MISS_CM - C_MISS_ASV * asv_params["p_miss_asv"]) - PI_NON * C_FA_ASV * asv_params["p_fa_asv"]
    c2 = C_FA_CM * PI_SPOOF * (1.0 - asv_params["p_miss_asv_spoof"])
    norm_factor = min(c1, c2)

    pos = cm_scores[cm_labels == 1]
    neg = cm_scores[cm_labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need both bonafide and spoof trials to compute t-DCF")

    thresholds = np.unique(np.concatenate([[-np.inf], np.unique(cm_scores), [np.inf]]))
    p_miss = np.array([_error_rates_at_threshold(pos, neg, t)[0] for t in thresholds])
    p_fa = np.array([_error_rates_at_threshold(pos, neg, t)[1] for t in thresholds])
    tdcf = c1 * p_miss + c2 * p_fa
    tdcf_norm = tdcf / norm_factor
    i = int(np.argmin(tdcf_norm))
    info = dict(asv_params)
    info.update({
        "C1": float(c1),
        "C2": float(c2),
        "norm_factor": float(norm_factor),
        "cm_threshold_opt": float(thresholds[i]),
        "p_miss_cm_at_opt": float(p_miss[i]),
        "p_fa_cm_at_opt": float(p_fa[i]),
    })
    return float(tdcf_norm[i]), info


def per_attack_eer(cm_scores, cm_labels, attacks):
    """EER of the pooled bonafide scores vs each individual attack id."""
    scores = np.asarray(cm_scores, dtype=np.float64)
    labels = np.asarray(cm_labels)
    bon = scores[labels == 1]
    out = {}
    for atk in sorted(set(attacks)):
        if atk == "-":
            continue
        mask = np.array([a == atk for a in attacks])
        spoof = scores[mask & (labels == 0)]
        if len(spoof) == 0:
            continue
        eer, _ = compute_eer(bon, spoof)
        out[atk] = {"eer": eer, "n_spoof": int(len(spoof))}
    return out


def bootstrap_eer_ci(scores, labels, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI over utterances for pooled EER."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(labels)
    eers = []
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    for _ in range(n_boot):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        q = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, q])
        eer, _ = compute_eer(scores[idx][labels[idx] == 1], scores[idx][labels[idx] == 0])
        eers.append(eer)
    lo, hi = np.percentile(eers, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"eer": float(np.mean(eers)), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": n_boot}


def full_metrics(cm_scores, cm_labels, attacks=None, asv=None, asv_params=None, bootstrap=False):
    """One-stop metrics dict used by src/evaluate.py."""
    scores = np.asarray(cm_scores, dtype=np.float64)
    labels = np.asarray(cm_labels)
    eer, eer_thr = compute_eer(scores[labels == 1], scores[labels == 0])
    min_tdcf, tdcf_info = compute_min_tdcf(scores, labels, asv=asv, asv_params=asv_params)
    out = {
        "eer": eer,
        "eer_threshold": eer_thr,
        "min_tDCF": min_tdcf,
        "tdcf_info": tdcf_info,
    }
    if attacks is not None:
        out["per_attack"] = per_attack_eer(scores, labels, attacks)
    if bootstrap:
        out["eer_bootstrap"] = bootstrap_eer_ci(scores, labels)
    return out
