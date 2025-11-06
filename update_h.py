import numpy as np
from .medians import weighted_median

def _l1_scale_for_pair(x: np.ndarray, w: np.ndarray, nonneg_s: bool = True, eps: float = 1e-12) -> float:
    """
    Solve s* = argmin_{s >= 0 (if nonneg_s)} ||x - s w||_1 via weighted median.
    If w is signed, weights are |w| and ratios are x / w.
    Returns s* (>=0 if nonneg_s else real >= 0 still projected by caller).
    """
    # Identify entries with non-negligible |w|
    mask = np.abs(w) > eps
    if not np.any(mask):
        return 0.0
    r = x[mask] / w[mask]
    p = np.abs(w[mask])
    s = weighted_median(r, p)
    if nonneg_s:
        s = max(0.0, s)
    return float(s)


def update_H_l1(X: np.ndarray,
                W: np.ndarray,
                enforce_W_nonneg: bool = True,
                eps: float = 1e-12):
    """
    L1orthNNLS:
    For each column x_j and each cluster k, compute s*_kj by weighted median and its L1 cost.
    Assign j to the k with minimal cost. Build H with H[k, j] = s*_kj and zeros elsewhere.
    Complexity (naive): O(n*k*m log m). Exploits zero weights where w_{dk}=0 implicitly.
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    m, n = X.shape
    _, k = W.shape
    H = np.zeros((k, n), dtype=float)

    # Precompute absolute W for weights
    absW = np.abs(W)

    for j in range(n):
        xj = X[:, j]
        best_k = 0
        best_cost = np.inf
        best_s = 0.0

        for kk in range(k):
            wk = W[:, kk]
            # scale
            s = _l1_scale_for_pair(xj, wk, nonneg_s=True, eps=eps)
            # cost
            cost = np.sum(np.abs(xj - s * wk))
            if cost < best_cost:
                best_cost = cost
                best_k = kk
                best_s = s

        H[best_k, j] = best_s

    return H
