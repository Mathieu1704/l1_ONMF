# medians.py
import numpy as np
from numba import njit

import numpy as np
from numba import njit

@njit(cache=True)
def weighted_median_numba(values, weights):
    """
    Calcule la médiane pondérée.
    Equivalent L1 de la moyenne pondérée utilisée en KL.
    """
    n = len(values)
    if n == 0: return 0.0
    
    # Tri conjoint
    idxs = np.argsort(values)
    v_sorted = values[idxs]
    w_sorted = weights[idxs]
    
    total_w = np.sum(w_sorted)
    if total_w <= 0: return 0.0
    
    half_w = 0.5 * total_w
    cum_w = 0.0
    
    for i in range(n):
        cum_w += w_sorted[i]
        if cum_w >= half_w:
            return v_sorted[i]
    return v_sorted[-1]

# Wrapper pour compatibilité si besoin, mais on appellera directement la version numba
def weighted_median(values, weights):
    return weighted_median_numba(np.asarray(values, dtype=float), np.asarray(weights, dtype=float))

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
