# l1_ONMF/update_h.py
import numpy as np
try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from .medians import weighted_median


def _is_sparse(X) -> bool:
    return sp is not None and sp.isspmatrix(X)


def _l1_scale_with_zeros(xnz: np.ndarray, wk_idx: np.ndarray, extra0_weight: float, eps: float) -> float:
    """
    Minimise s>=0 : sum_{idx} |x - s w| + sum_{not idx} |0 - s w|
    En sparse, les termes 'not idx' se résument à ajouter (ratio=0) avec weight=sum_{not idx}|w|.
    Les indices où w=0 ne contribuent pas à la médiane (mais contribuent au coût via |x|).
    """
    abs_w = np.abs(wk_idx)
    mask = abs_w > eps
    if not np.any(mask):
        return 0.0

    ratios = xnz[mask] / wk_idx[mask]
    weights = abs_w[mask]

    if extra0_weight > 0:
        ratios = np.concatenate([ratios, np.array([0.0])])
        weights = np.concatenate([weights, np.array([extra0_weight])])

    s = weighted_median(ratios, weights)
    return float(max(0.0, s))


def update_H_l1(X, W, enforce_W_nonneg: bool = True, eps: float = 1e-12):
    """
    Update H (hard clustering) exact pour l'objectif ℓ1.
    Pour chaque colonne j: choisir kk et s>=0 minimisant ||x_j - s w_kk||_1.
    - X peut être CSC/CSR ou dense
    - W est dense (m x k)
    """
    W = np.asarray(W, dtype=float)
    if enforce_W_nonneg:
        W = np.maximum(W, 0.0)

    m, k = W.shape
    absW_l1 = np.sum(np.abs(W), axis=0)  # ||w_k||_1

    if _is_sparse(X):
        Xc = X.tocsc()
        n = Xc.shape[1]
        H = np.zeros((k, n), dtype=float)
        indptr, indices, data = Xc.indptr, Xc.indices, Xc.data

        for j in range(n):
            a, b = indptr[j], indptr[j + 1]
            idx = indices[a:b]
            xnz = data[a:b]

            best_cost = np.inf
            best_k = 0
            best_s = 0.0

            for kk in range(k):
                wk_idx = W[idx, kk]                      # w sur les nnz de x
                extra0 = float(absW_l1[kk] - np.sum(np.abs(wk_idx)))  # poids exact des zéros
                if extra0 < 0:
                    extra0 = 0.0

                s = _l1_scale_with_zeros(xnz, wk_idx, extra0, eps)

                # coût exact : nnz + zéros
                cost_nz = float(np.sum(np.abs(xnz - s * wk_idx)))
                cost0 = float(s * extra0)
                cost = cost_nz + cost0

                if cost < best_cost:
                    best_cost = cost
                    best_k = kk
                    best_s = s

            H[best_k, j] = best_s

        return H

    # dense
    Xd = np.asarray(X, dtype=float)
    m2, n = Xd.shape
    assert m2 == m
    H = np.zeros((k, n), dtype=float)

    for j in range(n):
        xj = Xd[:, j]
        idx = np.where(xj != 0)[0]
        xnz = xj[idx]

        best_cost = np.inf
        best_k = 0
        best_s = 0.0

        for kk in range(k):
            wk_idx = W[idx, kk]
            extra0 = float(absW_l1[kk] - np.sum(np.abs(wk_idx)))
            if extra0 < 0:
                extra0 = 0.0

            s = _l1_scale_with_zeros(xnz, wk_idx, extra0, eps)

            cost_nz = float(np.sum(np.abs(xnz - s * wk_idx)))
            cost0 = float(s * extra0)
            cost = cost_nz + cost0

            if cost < best_cost:
                best_cost = cost
                best_k = kk
                best_s = s

        H[best_k, j] = best_s

    return H
