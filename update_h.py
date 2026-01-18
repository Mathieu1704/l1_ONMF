# l1_ONMF/update_h.py
import numpy as np
from numba import njit, prange

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from .medians import weighted_median_numba


def _is_sparse(X) -> bool:
    return sp is not None and sp.isspmatrix(X)


@njit(cache=True)
def _solve_h_col(x_data, x_indices, W, absW_l1, k, eps, zero_weight):
    """
    Résout H pour une colonne j donnée avec objectif pondéré par zero_weight (gamma).

    Remarque:
    - Cette routine est "naturelle" pour le cas NMF (W >= 0). Si W contient des valeurs négatives,
      elle reste calculable mais les garanties de correction (liées aux formules de médiane pondérée)
      ne sont plus assurées. On gère ce point via une garde dans l'algorithme principal.
    """
    best_cost = np.inf
    best_k = 0
    best_s = 0.0

    for kk in range(k):
        # Somme |W_i,kk| sur les indices observés
        sum_abs_w_idx = 0.0
        for t in range(len(x_indices)):
            row_idx = x_indices[t]
            val = W[row_idx, kk]
            if val < 0.0:
                sum_abs_w_idx += -val
            else:
                sum_abs_w_idx += val

        # Contribution des zéros implicites : absW_l1 - sum_abs_w_idx
        extra0 = absW_l1[kk] - sum_abs_w_idx
        if extra0 < 0.0:
            extra0 = 0.0

        # Compte des points utiles (|w| > eps) pour les ratios x/w
        count = 0
        for t in range(len(x_indices)):
            row_idx = x_indices[t]
            w_val = W[row_idx, kk]
            if w_val < 0.0:
                if -w_val > eps:
                    count += 1
            else:
                if w_val > eps:
                    count += 1

        # +1 pour pseudo-point 0 si extra0 > 0
        ratios = np.empty(count + 1, dtype=np.float64)
        weights = np.empty(count + 1, dtype=np.float64)

        ptr = 0
        for t in range(len(x_indices)):
            row_idx = x_indices[t]
            w_val = W[row_idx, kk]
            aw = -w_val if (w_val < 0.0) else w_val
            if aw > eps:
                ratios[ptr] = x_data[t] / w_val
                weights[ptr] = aw
                ptr += 1

        # pseudo-point 0 (zéros implicites) pondéré par gamma=zero_weight
        if extra0 > 0.0:
            ratios[ptr] = 0.0
            weights[ptr] = extra0 * zero_weight
            ptr += 1

        s = 0.0
        if ptr > 0:
            s = weighted_median_numba(ratios[:ptr], weights[:ptr])
            if s < 0.0:
                s = 0.0

        # coût : somme |x - s*w| sur non-zéros + gamma * s * extra0
        cost_nz = 0.0
        for t in range(len(x_indices)):
            row_idx = x_indices[t]
            cost_nz += abs(x_data[t] - s * W[row_idx, kk])

        current_cost = cost_nz + s * extra0 * zero_weight

        if current_cost < best_cost:
            best_cost = current_cost
            best_k = kk
            best_s = s

    return best_k, best_s


@njit(parallel=True, cache=True)
def _update_H_csc_numba(data, indices, indptr, n, W, absW_l1, k, eps, zero_weight):
    H_res = np.zeros((k, n), dtype=np.float64)
    for j in prange(n):
        start = indptr[j]
        end = indptr[j + 1]

        # Colonne vide : on garde une valeur epsilon pour éviter un doc "non assigné"
        if start == end:
            H_res[0, j] = 1e-9
            continue

        x_d = data[start:end]
        x_i = indices[start:end]
        bk, bs = _solve_h_col(x_d, x_i, W, absW_l1, k, eps, zero_weight)

        if bs < 1e-9:
            bs = 1e-9
        H_res[bk, j] = bs
    return H_res


@njit(parallel=True, cache=True)
def _update_H_dense_numba(X, n, m, W, absW_l1, k, eps, zero_weight):
    H_res = np.zeros((k, n), dtype=np.float64)
    all_indices = np.arange(m)

    for j in prange(n):
        x_d = X[:, j]
        bk, bs = _solve_h_col(x_d, all_indices, W, absW_l1, k, eps, zero_weight)
        if bs < 1e-9:
            bs = 1e-9
        H_res[bk, j] = bs
    return H_res


def update_H_l1(
    X,
    W,
    enforce_W_nonneg: bool = True,
    eps: float = 1e-12,
    zero_weight: float = 1.0,
):
    """
    Update H (hard assignment + optimal scaling via médiane pondérée),
    avec gestion des zéros implicites pondérée par gamma=zero_weight.

    Note: si enforce_W_nonneg=False et W contient des négatifs, la routine reste
    exécutable mais n'a plus les garanties théoriques associées au cas NMF.
    La garde/warning se fait côté `alternating_l1_onmf`.
    """
    W = np.asarray(W, dtype=float)

    if enforce_W_nonneg:
        W = np.maximum(W, 0.0)

    m, k = W.shape
    absW_l1 = np.sum(np.abs(W), axis=0)
    zw = float(zero_weight)

    if _is_sparse(X):
        Xcsc = X.tocsc()
        n = Xcsc.shape[1]
        return _update_H_csc_numba(
            Xcsc.data.astype(np.float64),
            Xcsc.indices.astype(np.int32),
            Xcsc.indptr.astype(np.int32),
            n,
            W,
            absW_l1,
            k,
            float(eps),
            zw,
        )
    else:
        Xd = np.asarray(X, dtype=float, order="F")
        n = Xd.shape[1]
        return _update_H_dense_numba(
            Xd,
            n,
            m,
            W,
            absW_l1,
            k,
            float(eps),
            zw,
        )
