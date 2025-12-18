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

def update_W_l1(X, H: np.ndarray, enforce_W_nonneg: bool = True, eps: float = 1e-12):
    """
    Update W pour L1-ONMF.
    - Si X est CSR sparse: pour chaque ligne d, on ne regarde que les nnz de cette ligne
      et on agrège la masse des zéros via zero_weight = sum(s_j) - sum(s_j sur nnz).
    - Sinon: fallback dense (ton ancienne version).
    """
    H = np.asarray(H, dtype=float)
    k, n = H.shape
    m = X.shape[0]
    W = np.zeros((m, k), dtype=float)

    if sp is not None and sp.isspmatrix(X):
        X = X.tocsr()

        # Précompute par cluster
        cols_by_cluster = []
        s_by_cluster = []
        sum_s = np.zeros(k, dtype=float)

        for i in range(k):
            cols = np.where(H[i, :] > eps)[0]
            s = H[i, cols]
            cols_by_cluster.append(cols)
            s_by_cluster.append(s)
            sum_s[i] = float(np.sum(s))

        # Pour filtrer vite si un nnz appartient au cluster i, on construit des masques bool
        in_cluster = [np.zeros(n, dtype=bool) for _ in range(k)]
        s_at_col = [np.zeros(n, dtype=float) for _ in range(k)]
        for i in range(k):
            cols = cols_by_cluster[i]
            in_cluster[i][cols] = True
            s_at_col[i][cols] = H[i, cols]

        for d in range(m):
            row_start, row_end = X.indptr[d], X.indptr[d + 1]
            row_cols = X.indices[row_start:row_end]
            row_vals = X.data[row_start:row_end].astype(float)

            for i in range(k):
                if sum_s[i] <= eps:
                    continue  # cluster vide

                # On ne garde que les nnz de cette ligne qui sont dans le cluster
                mask = in_cluster[i][row_cols]
                if not np.any(mask):
                    # Tout est zéro sur cette ligne pour ce cluster -> médiane de {0} => 0
                    w_med = 0.0
                else:
                    cols_nz = row_cols[mask]
                    x_nz = row_vals[mask]
                    s_nz = s_at_col[i][cols_nz]  # poids et dénominateur

                    # élimine s trop petits
                    good = s_nz > eps
                    x_nz = x_nz[good]
                    s_nz = s_nz[good]

                    if x_nz.size == 0:
                        w_med = 0.0
                    else:
                        ratios = x_nz / s_nz
                        weights = s_nz
                        w0 = float(sum_s[i] - float(np.sum(weights)))
                        w_med = weighted_median_with_zero(ratios, weights, w0)

                if enforce_W_nonneg and w_med < 0:
                    w_med = 0.0
                W[d, i] = float(w_med)

        return W

    # ---- Fallback dense ----
    X = np.asarray(X, dtype=float)

    for i in range(k):
        s = H[i, :]
        mask_j = s > eps
        if not np.any(mask_j):
            continue
        s_sel = s[mask_j]
        X_sel = X[:, mask_j]

        for d in range(m):
            ratios = X_sel[d, :] / s_sel
            w_med = weighted_median(ratios, s_sel)
            if enforce_W_nonneg:
                w_med = max(0.0, float(w_med))
            W[d, i] = w_med

    return W
