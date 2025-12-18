import numpy as np

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted median on 1D arrays (values, weights >= 0), O(t log t) by sorting.
    If total weight is zero, returns 0.0.
    """
    if values.size == 0:
        return 0.0
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = w > 0
    if not np.any(mask):
        return 0.0
    v = v[mask]
    w = w[mask]
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    csum = np.cumsum(w)
    half = 0.5 * csum[-1]
    idx = np.searchsorted(csum, half, side="right")
    idx = min(idx, v.size - 1)
    return float(v[idx])


def median(values: np.ndarray) -> float:
    """Unweighted median (fallback), returns 0.0 on empty."""
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute num/den with protection; entries with |den|<eps are ignored upstream."""
    d = np.asarray(den, dtype=float)
    n = np.asarray(num, dtype=float)
    out = np.empty_like(n, dtype=float)
    mask = np.abs(d) >= eps
    out[mask] = n[mask] / d[mask]
    out[~mask] = 0.0
    return out

def weighted_median_with_zero(values: np.ndarray, weights: np.ndarray, zero_weight: float) -> float:
    """
    Médiane pondérée de (values, weights) + un point additionnel value=0 avec poids zero_weight.
    Évite de parcourir tous les zéros explicitement.
    """
    if zero_weight > 0:
        v = np.concatenate([np.asarray(values, float), np.array([0.0])])
        w = np.concatenate([np.asarray(weights, float), np.array([float(zero_weight)])])
        return weighted_median(v, w)
    else:
        return weighted_median(values, weights)
