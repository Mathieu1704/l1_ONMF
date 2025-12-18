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
def _solve_h_col(x_data, x_indices, W, absW_l1, k, eps):
    """Résout H pour une colonne j donnée (logique commune sparse/dense)"""
    best_cost = np.inf
    best_k = 0
    best_s = 0.0

    # On teste chaque cluster k
    for kk in range(k):
        # Récupération manuelle des valeurs de W pour les indices concernés
        # W est (m, k). x_indices est de taille variable.
        
        sum_abs_w_idx = 0.0
        # On calcule d'abord la somme pour extra0
        for idx_i in range(len(x_indices)):
            row_idx = x_indices[idx_i]
            val = W[row_idx, kk]
            sum_abs_w_idx += abs(val)
            
        extra0 = absW_l1[kk] - sum_abs_w_idx
        if extra0 < 0: extra0 = 0.0
        
        # Construction des inputs pour la médiane pondérée
        # On fait une première passe pour compter les éléments non-nuls (pour allocation)
        count = 0
        for idx_i in range(len(x_indices)):
            row_idx = x_indices[idx_i]
            if abs(W[row_idx, kk]) > eps:
                count += 1
        
        # Allocation (+1 pour le potentiel extra0)
        ratios = np.empty(count + 1, dtype=np.float64)
        weights = np.empty(count + 1, dtype=np.float64)
        
        ptr = 0
        for idx_i in range(len(x_indices)):
            row_idx = x_indices[idx_i]
            w_val = W[row_idx, kk]
            if abs(w_val) > eps:
                ratios[ptr] = x_data[idx_i] / w_val
                weights[ptr] = abs(w_val)
                ptr += 1
        
        if extra0 > 0:
            ratios[ptr] = 0.0
            weights[ptr] = extra0
            ptr += 1
            
        s = 0.0
        if ptr > 0:
            s = weighted_median_numba(ratios[:ptr], weights[:ptr])
            if s < 0: s = 0.0
            
        # Calcul du coût L1
        # cost = sum |x - s*w| + s * extra0
        cost_nz = 0.0
        for idx_i in range(len(x_indices)):
            row_idx = x_indices[idx_i]
            cost_nz += abs(x_data[idx_i] - s * W[row_idx, kk])
            
        current_cost = cost_nz + s * extra0
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_k = kk
            best_s = s
            
    return best_k, best_s

@njit(parallel=True, cache=True)
def _update_H_csc_numba(data, indices, indptr, n, W, absW_l1, k, eps):
    H_res = np.zeros((k, n), dtype=np.float64)
    for j in prange(n):
        start = indptr[j]
        end = indptr[j+1]
        if start == end: continue
        x_d = data[start:end]
        x_i = indices[start:end]
        bk, bs = _solve_h_col(x_d, x_i, W, absW_l1, k, eps)
        # Fix anti-mort
        if bs < 1e-9: bs = 1e-9
        H_res[bk, j] = bs
    return H_res

@njit(parallel=True, cache=True)
def _update_H_dense_numba(X, n, m, W, absW_l1, k, eps):
    H_res = np.zeros((k, n), dtype=np.float64)
    # Indices constants [0, 1, ..., m-1] pour le cas dense
    all_indices = np.arange(m) 
    
    for j in prange(n):
        # Colonne dense
        x_d = X[:, j] # Copie ou vue
        bk, bs = _solve_h_col(x_d, all_indices, W, absW_l1, k, eps)
        if bs < 1e-9: bs = 1e-9
        H_res[bk, j] = bs
    return H_res

def update_H_l1(X, W, enforce_W_nonneg: bool = True, eps: float = 1e-12):
    W = np.asarray(W, dtype=float)
    if enforce_W_nonneg:
        W = np.maximum(W, 0.0)
    
    m, k = W.shape
    absW_l1 = np.sum(np.abs(W), axis=0)
    
    if _is_sparse(X):
        Xcsc = X.tocsc()
        n = Xcsc.shape[1]
        return _update_H_csc_numba(
            Xcsc.data.astype(np.float64),
            Xcsc.indices.astype(np.int32),
            Xcsc.indptr.astype(np.int32),
            n, W, absW_l1, k, eps
        )
    else:
        # CAS DENSE
        Xd = np.asarray(X, dtype=float, order='F') # order F pour accès colonne rapide
        n = Xd.shape[1]
        return _update_H_dense_numba(Xd, n, m, W, absW_l1, k, eps)