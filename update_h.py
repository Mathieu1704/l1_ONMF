import numpy as np
<<<<<<< Updated upstream
from .medians import weighted_median
=======

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

try:
    from .medians import weighted_median_with_zero, weighted_median
except ImportError:
    from medians import weighted_median_with_zero, weighted_median

>>>>>>> Stashed changes

def update_H_l1(X, W: np.ndarray, enforce_W_nonneg: bool = True, eps: float = 1e-12):
    """
    Update H pour L1-ONMF.
    - Si X est CSC sparse: utilise uniquement les nnz de chaque colonne + masse de zéros agrégée.
    - Sinon: fallback dense.
    """
    W = np.asarray(W, dtype=float)
    m, k = W.shape

    if sp is not None and sp.isspmatrix(X):
        X = X.tocsc()
        _, n = X.shape
        H = np.zeros((k, n), dtype=float)

        # Précompute pour chaque cluster:
        # sum_w_pos = sum_{i: w_i > eps} w_i  (utile pour la médiane)
        # sum_w_all = sum_i w_i              (utile pour le coût avec x=0)
        W_pos = W > eps
        sum_w_pos = np.sum(W * W_pos, axis=0)  # (k,)
        sum_w_all = np.sum(W, axis=0)          # (k,)

        for j in range(n):
            start, end = X.indptr[j], X.indptr[j + 1]
            idx = X.indices[start:end]      # indices non nuls (même si valeur petite)
            xnz = X.data[start:end].astype(float)

            best_cost = np.inf
            best_k = 0
            best_s = 0.0

            for kk in range(k):
                wk = W[:, kk]

                # Sur indices non nuls, on prend wk[idx]
                wk_idx = wk[idx]
                mask = wk_idx > eps

                # ratios/poids pour les positions où wk>0
                ratios = xnz[mask] / wk_idx[mask]
                weights = wk_idx[mask]

                # masse de zéros (x=0) pour les positions où wk>0
                # total poids pour la médiane = sum_w_pos[kk]
                w_nz_pos = float(np.sum(weights))
                w0 = float(sum_w_pos[kk] - w_nz_pos)

                # scale s = médiane pondérée + projection >=0
                s = weighted_median_with_zero(ratios, weights, w0)
                if s < 0:
                    s = 0.0

                # coût L1:
                # cost_nz = sum_{i in nz} |x_i - s*w_i|
                cost_nz = float(np.sum(np.abs(xnz - s * wk_idx)))
                # cost_zero = sum_{i: x=0} |0 - s*w_i| = s * sum_{i: x=0} w_i
                # sum_{i: x=0} w_i = sum_w_all - sum_{i in nz} w_i
                cost_zero = float(s * (sum_w_all[kk] - float(np.sum(wk_idx))))
                cost = cost_nz + cost_zero

                if cost < best_cost:
                    best_cost = cost
                    best_k = kk
                    best_s = s

            H[best_k, j] = best_s

        return H

    # ---- Fallback dense (ton ancienne version, simplifiée) ----
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    H = np.zeros((k, n), dtype=float)

    for j in range(n):
        xj = X[:, j]
        best_k = 0
        best_cost = np.inf
        best_s = 0.0

        for kk in range(k):
            wk = W[:, kk]
            mask = np.abs(wk) > eps
            if not np.any(mask):
                s = 0.0
            else:
                r = xj[mask] / wk[mask]
                p = np.abs(wk[mask])
                s = weighted_median(r, p)
                s = max(0.0, float(s))

            cost = float(np.sum(np.abs(xj - s * wk)))
            if cost < best_cost:
                best_cost = cost
                best_k = kk
                best_s = s

        H[best_k, j] = best_s

    return H
