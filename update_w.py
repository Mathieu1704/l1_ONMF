# l1_ONMF/update_w.py
import numpy as np
from numba import njit, prange
try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from .medians import weighted_median_numba


def _is_sparse(X) -> bool:
    return sp is not None and sp.isspmatrix(X)

@njit(parallel=True, cache=True)
def _update_W_csr_numba(data, indices, indptr, m, k, assign, s, enforce_nonneg, sum_s, eps, zero_weight):
    W = np.zeros((m, k), dtype=np.float64)
    for d in prange(m):
        start_idx = indptr[d]
        end_idx = indptr[d+1]
        if start_idx == end_idx:
            continue

        row_cols = indices[start_idx:end_idx]
        row_vals = data[start_idx:end_idx]

        for cluster_id in range(k):
            count = 0
            curr_sum_weights = 0.0

            # Passe 1 : compter les non-nuls dans ce cluster
            for i in range(len(row_cols)):
                doc_idx = row_cols[i]
                if assign[doc_idx] == cluster_id:
                    if s[doc_idx] > eps:
                        curr_sum_weights += s[doc_idx]
                        count += 1

            if count == 0 and sum_s[cluster_id] <= eps:
                continue

            vals_med = np.empty(count + 1, dtype=np.float64)
            wgts_med = np.empty(count + 1, dtype=np.float64)

            ptr = 0
            # Passe 2 : ratios présents
            for i in range(len(row_cols)):
                doc_idx = row_cols[i]
                if assign[doc_idx] == cluster_id:
                    si = s[doc_idx]
                    if si > eps:
                        vals_med[ptr] = row_vals[i] / si
                        wgts_med[ptr] = si
                        ptr += 1

            # Zéros implicites : point à 0 avec poids réduit
            extra0 = sum_s[cluster_id] - curr_sum_weights
            if extra0 > 1e-14:
                vals_med[ptr] = 0.0
                wgts_med[ptr] = extra0 * zero_weight
                ptr += 1

            if ptr > 0:
                val = weighted_median_numba(vals_med[:ptr], wgts_med[:ptr])
                if enforce_nonneg and val < 0:
                    val = 0.0
                W[d, cluster_id] = val

    return W


@njit(parallel=True, cache=True)
def _update_W_dense_numba(X, m, n, k, assign, s, enforce_nonneg, eps, zero_weight):
    W = np.zeros((m, k), dtype=np.float64)
    gamma = zero_weight

    for d in prange(m):
        row_vals = X[d, :]

        for cluster_id in range(k):
            # Compter combien d'éléments utiles dans ce cluster (s>eps)
            count = 0
            for j in range(n):
                if assign[j] == cluster_id and s[j] > eps:
                    count += 1
            if count == 0:
                continue

            vals_med = np.empty(count, dtype=np.float64)
            wgts_med = np.empty(count, dtype=np.float64)

            ptr = 0
            for j in range(n):
                if assign[j] == cluster_id:
                    si = s[j]
                    if si > eps:
                        xdj = row_vals[j]
                        vals_med[ptr] = xdj / si
                        # poids = si si x!=0, sinon gamma*si
                        wgts_med[ptr] = si if xdj != 0.0 else (gamma * si)
                        ptr += 1

            if ptr > 0:
                val = weighted_median_numba(vals_med[:ptr], wgts_med[:ptr])
                if enforce_nonneg and val < 0.0:
                    val = 0.0
                W[d, cluster_id] = val

    return W

def update_W_l1(X, H, enforce_W_nonneg=True, eps=1e-12, zero_weight: float = 0.05):
    H = np.asarray(H, dtype=float)
    k, n = H.shape
    assign = np.argmax(H, axis=0).astype(np.int32)
    s = H[assign, np.arange(n)].astype(np.float64)
    sum_s = np.bincount(assign, weights=s, minlength=k).astype(np.float64)

    zw = float(zero_weight)

    if _is_sparse(X):
        Xcsr = X.tocsr()
        data = Xcsr.data.astype(np.float64)
        indices = Xcsr.indices.astype(np.int32)
        indptr = Xcsr.indptr.astype(np.int32)
        m = Xcsr.shape[0]
        return _update_W_csr_numba(data, indices, indptr, m, k, assign, s, enforce_W_nonneg, sum_s, eps, zw)
    else:
        Xd = np.asarray(X, dtype=float)
        m = Xd.shape[0]
        return _update_W_dense_numba(Xd, m, n, k, assign, s, enforce_W_nonneg, eps, zw)
