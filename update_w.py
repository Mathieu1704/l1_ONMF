# l1_ONMF/update_w.py
import numpy as np
try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from .medians import weighted_median


def _is_sparse(X) -> bool:
    return sp is not None and sp.isspmatrix(X)


def update_W_l1(X, H: np.ndarray, enforce_W_nonneg: bool = True, eps: float = 1e-12):
    """
    Update W exact pour ℓ1 avec hard clustering.
    Pour chaque mot d et cluster i:
        w_{d,i} = weighted median des ratios (x_{d,j}/s_j) avec poids s_j,
    en incluant les docs où x_{d,j}=0 via un point ratio=0 de poids (sum_s[i] - sum_{present} s_j).

    Si enforce_W_nonneg, on projette w>=0.
    """
    H = np.asarray(H, dtype=float)
    k, n = H.shape
    assign = np.argmax(H, axis=0).astype(int)
    s = H[assign, np.arange(n)].astype(float)

    # somme des s par cluster
    sum_s = np.bincount(assign, weights=s, minlength=k)

    if _is_sparse(X):
        Xr = X.tocsr().astype(float)
        m = Xr.shape[0]
        W = np.zeros((m, k), dtype=float)

        indptr, indices, data = Xr.indptr, Xr.indices, Xr.data

        for d in range(m):
            a, b = indptr[d], indptr[d + 1]
            if a == b:
                continue

            cols = indices[a:b]
            vals = data[a:b]

            cl = assign[cols]
            s_cols = s[cols]

            # On ignore les docs où s=0 (n’influencent pas w)
            valid = s_cols > eps
            if not np.any(valid):
                continue

            cols = cols[valid]
            vals = vals[valid]
            cl = cl[valid]
            s_cols = s_cols[valid]

            # Optimisation : on ne calcule que pour les clusters présents dans cette ligne
            for i in np.unique(cl):
                mask = (cl == i)
                weights = s_cols[mask]
                if weights.size == 0:
                    continue

                ratios = vals[mask] / weights
                # poids exact des zéros (docs du cluster i où le mot d est absent)
                extra0 = float(sum_s[i] - np.sum(weights))
                if extra0 > 0:
                    ratios = np.concatenate([ratios, np.array([0.0])])
                    weights = np.concatenate([weights, np.array([extra0])])

                w = float(weighted_median(ratios, weights))
                if enforce_W_nonneg:
                    w = max(0.0, w)
                W[d, i] = w

            # Clusters absents => tous x=0 => médiane = 0 automatiquement (déjà 0)

        return W

    # dense fallback
    Xd = np.asarray(X, dtype=float)
    m, _ = Xd.shape
    W = np.zeros((m, k), dtype=float)

    for i in range(k):
        mask_j = (assign == i) & (s > eps)
        if not np.any(mask_j):
            continue
        s_sel = s[mask_j]
        X_sel = Xd[:, mask_j]

        for d in range(m):
            col = X_sel[d, :]
            nz = col != 0
            weights_nz = s_sel[nz]
            ratios_nz = col[nz] / weights_nz if np.any(nz) else np.array([], dtype=float)

            extra0 = float(np.sum(s_sel) - np.sum(weights_nz))
            ratios = ratios_nz
            weights = weights_nz
            if extra0 > 0:
                ratios = np.concatenate([ratios, np.array([0.0])])
                weights = np.concatenate([weights, np.array([extra0])])

            w = float(weighted_median(ratios, weights)) if weights.size > 0 else 0.0
            if enforce_W_nonneg:
                w = max(0.0, w)
            W[d, i] = w

    return W
