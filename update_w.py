# l1_ONMF/update_w.py
import numpy as np
from numba import njit, prange
try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from .medians import weighted_median_numba

# --- PARAMETRE DE REGULARISATION ---
# C'est la "sauce secrète" pour le Sparse L1.
# 0.05 signifie que les zéros "tirent" 20 fois moins fort sur la médiane que les vraies valeurs.
# Cela permet au centroïde de s'aligner sur les mots présents.
ZERO_WEIGHT = 0.05 
# -----------------------------------

def _is_sparse(X) -> bool:
    return sp is not None and sp.isspmatrix(X)

@njit(parallel=True, cache=True)
def _update_W_csr_numba(data, indices, indptr, m, k, assign, s, enforce_nonneg, sum_s, eps):
    W = np.zeros((m, k), dtype=np.float64)
    for d in prange(m):
        start_idx = indptr[d]
        end_idx = indptr[d+1]
        if start_idx == end_idx: continue
            
        row_cols = indices[start_idx:end_idx]
        row_vals = data[start_idx:end_idx]
        
        for cluster_id in range(k):
            count = 0
            curr_sum_weights = 0.0
            
            # Passe 1 : on compte les éléments non-nuls pour ce cluster
            for i in range(len(row_cols)):
                doc_idx = row_cols[i]
                if assign[doc_idx] == cluster_id:
                    if s[doc_idx] > eps:
                        curr_sum_weights += s[doc_idx]
                        count += 1
            
            if count == 0 and sum_s[cluster_id] <= eps: continue

            # Allocation dynamique
            vals_med = np.empty(count + 1, dtype=np.float64)
            wgts_med = np.empty(count + 1, dtype=np.float64)
            
            ptr = 0
            # Passe 2 : on remplit avec les valeurs présentes
            for i in range(len(row_cols)):
                doc_idx = row_cols[i]
                if assign[doc_idx] == cluster_id:
                    si = s[doc_idx]
                    if si > eps:
                        vals_med[ptr] = row_vals[i] / si
                        wgts_med[ptr] = si
                        ptr += 1
            
            # REGULARISATION DES ZEROS (Les documents du cluster qui n'ont pas ce mot)
            extra0 = sum_s[cluster_id] - curr_sum_weights
            if extra0 > 1e-14:
                vals_med[ptr] = 0.0
                # On applique le poids réduit aux zéros pour éviter l'effondrement
                wgts_med[ptr] = extra0 * ZERO_WEIGHT 
                ptr += 1
            
            if ptr > 0:
                val = weighted_median_numba(vals_med[:ptr], wgts_med[:ptr])
                if enforce_nonneg and val < 0: val = 0.0
                W[d, cluster_id] = val
    return W

@njit(parallel=True, cache=True)
def _update_W_dense_numba(X, m, n, k, assign, s, enforce_nonneg, sum_s, eps):
    W = np.zeros((m, k), dtype=np.float64)
    for d in prange(m):
        row_vals = X[d, :] 
        for cluster_id in range(k):
            count = 0
            curr_sum_weights = 0.0
            # Passe 1
            for j in range(n):
                if assign[j] == cluster_id:
                    if s[j] > eps:
                        curr_sum_weights += s[j]
                        count += 1
            if count == 0 and sum_s[cluster_id] <= eps: continue

            vals_med = np.empty(count + 1, dtype=np.float64)
            wgts_med = np.empty(count + 1, dtype=np.float64)
            ptr = 0
            # Passe 2
            for j in range(n):
                if assign[j] == cluster_id:
                    si = s[j]
                    if si > eps:
                        vals_med[ptr] = row_vals[j] / si
                        wgts_med[ptr] = si
                        ptr += 1
            
            # REGULARISATION DES ZEROS
            extra0 = sum_s[cluster_id] - curr_sum_weights
            if extra0 > 1e-14:
                vals_med[ptr] = 0.0
                wgts_med[ptr] = extra0 * ZERO_WEIGHT 
                ptr += 1
                
            if ptr > 0:
                val = weighted_median_numba(vals_med[:ptr], wgts_med[:ptr])
                if enforce_nonneg and val < 0: val = 0.0
                W[d, cluster_id] = val
    return W

def update_W_l1(X, H, enforce_W_nonneg=True, eps=1e-12):
    H = np.asarray(H, dtype=float)
    k, n = H.shape
    assign = np.argmax(H, axis=0).astype(np.int32)
    s = H[assign, np.arange(n)].astype(np.float64)
    sum_s = np.bincount(assign, weights=s, minlength=k).astype(np.float64)

    if _is_sparse(X):
        # Utilisation de CSR (Compressed Sparse Row) car on accède par ligne (mot)
        Xcsr = X.tocsr() 
        data = Xcsr.data.astype(np.float64)
        indices = Xcsr.indices.astype(np.int32)
        indptr = Xcsr.indptr.astype(np.int32)
        m = Xcsr.shape[0]
        return _update_W_csr_numba(data, indices, indptr, m, k, assign, s, enforce_W_nonneg, sum_s, eps)
    else:
        # Cas dense
        Xd = np.asarray(X, dtype=float)
        m = Xd.shape[0]
        return _update_W_dense_numba(Xd, m, n, k, assign, s, enforce_W_nonneg, sum_s, eps)