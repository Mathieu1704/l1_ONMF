import numpy as np
try:
    from .medians import weighted_median
except ImportError:
    from medians import weighted_median


def update_W_l1(X: np.ndarray,
                H: np.ndarray,
                enforce_W_nonneg: bool = True,
                eps: float = 1e-12):
    """
    For each cluster i and coordinate d:
      w_{di} <- w-median of { x_{dj} / s_j } with weights s_j over j in K_i (s_j>0).
    If enforce_W_nonneg, project negative entries to 0.
    """
    X = np.asarray(X, dtype=float)
    H = np.asarray(H, dtype=float)
    m, n = X.shape
    k = H.shape[0]
    W = np.zeros((m, k), dtype=float)

    for i in range(k):
        s = H[i, :]
        mask_j = s > eps
        if not np.any(mask_j):
            # empty cluster; let caller handle reinit if desired
            continue
        s_sel = s[mask_j]
        X_sel = X[:, mask_j]  # shape (m, |K_i|)

        # For each coordinate d, r_j = x_{dj}/s_j with weight s_j
        # Vectorized pattern: process row by row
        for d in range(m):
            col = X_sel[d, :]            # shape (|K_i|,)
            ratios = col / s_sel
            w_med = weighted_median(ratios, s_sel)
            if enforce_W_nonneg:
                w_med = max(0.0, w_med)
            W[d, i] = w_med

    return W
