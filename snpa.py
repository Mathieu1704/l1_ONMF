# l1_onmf/snpa.py
from __future__ import annotations

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


def _is_sparse(X) -> bool:
    return sp is not None and sp.isspmatrix(X)


def _col_l1_normalize_inplace_csc(Xcsc, eps: float = 1e-12):
    """
    Normalise les colonnes d'une CSC: x_j <- x_j / (sum(x_j)+eps).
    Retourne les sommes originales (pour info).
    """
    n = Xcsc.shape[1]
    col_sums = np.asarray(Xcsc.sum(axis=0)).ravel()
    col_sums = np.maximum(col_sums, eps)
    inv = 1.0 / col_sums

    # scale data in-place par colonne
    indptr = Xcsc.indptr
    data = Xcsc.data
    for j in range(n):
        a, b = indptr[j], indptr[j + 1]
        if a != b:
            data[a:b] *= inv[j]
    return col_sums


def _col_l2_norms_sq_csc(Xcsc) -> np.ndarray:
    """
    ||x_j||_2^2 pour chaque colonne d'une CSC (sans densifier).
    """
    n = Xcsc.shape[1]
    indptr = Xcsc.indptr
    data = Xcsc.data
    out = np.zeros(n, dtype=float)
    for j in range(n):
        a, b = indptr[j], indptr[j + 1]
        if a != b:
            out[j] = float(np.dot(data[a:b], data[a:b]))
    return out


def nnls_hals(AtA: np.ndarray, AtB: np.ndarray, n_iter: int = 50, eps: float = 1e-12) -> np.ndarray:
    """
    Résout (approx) NNLS en bloc: min_{H>=0} ||X - W H||_F^2
    en utilisant HALS vectorisé, où:
      AtA = W^T W  (k x k)
      AtB = W^T X  (k x n)
    Retour: H (k x n) >= 0
    """
    k, n = AtB.shape
    H = np.maximum(0.0, AtB / (np.diag(AtA)[:, None] + eps))

    diag = np.diag(AtA).copy()
    diag[diag < eps] = eps

    for _ in range(n_iter):
        # un sweep HALS
        for i in range(k):
            # grad component: (AtA[i,:] @ H) - AtB[i,:]
            AiH = AtA[i, :] @ H  # (n,)
            H[i, :] = np.maximum(0.0, H[i, :] + (AtB[i, :] - AiH) / diag[i])
    return H


def snpa(
    X,
    r: int,
    *,
    normalize: str = "l1",      # "l1" ou "none"
    nnls_iter: int = 50,
    eps: float = 1e-12,
    seed: int | None = None,
    verbose: bool = False,
):
    """
    SNPA (Successive Nonnegative Projection Algorithm)
    Sélectionne r colonnes de X (indices) pour initialiser W = X[:, K].

    Hypothèses utiles:
      - X >= 0 (docs/HSI ok)
      - r petit (4..50 typiquement)

    Retour:
      K: np.ndarray shape (r,) indices sélectionnés
      W: np.ndarray shape (m, r) = X[:,K] (dense)
      info: dict (résidus, etc.)
    """
    if r <= 0:
        raise ValueError("r doit être >= 1")

    rng = np.random.default_rng(seed)

    # --- Prépare X (sparse-friendly) ---
    if _is_sparse(X):
        Xcsc = X.tocsc(copy=True).astype(float)
        m, n = Xcsc.shape
        if (Xcsc.data < 0).any():
            raise ValueError("SNPA nécessite X >= 0 (valeurs négatives détectées).")
        if normalize == "l1":
            _ = _col_l1_normalize_inplace_csc(Xcsc, eps=eps)
        x_norm2 = _col_l2_norms_sq_csc(Xcsc)
    else:
        Xd = np.asarray(X, dtype=float)
        if np.any(Xd < 0):
            raise ValueError("SNPA nécessite X >= 0 (valeurs négatives détectées).")
        m, n = Xd.shape
        if normalize == "l1":
            col_sums = Xd.sum(axis=0)
            col_sums[col_sums < eps] = 1.0
            Xd = Xd / col_sums[None, :]
        Xcsc = Xd  # pour unifier la suite
        x_norm2 = np.sum(Xd * Xd, axis=0)

    if r > n:
        raise ValueError(f"r={r} > n={n} impossible.")

    # --- Sélection ---
    K = []
    resid_trace = []

    # 1) premier pivot = colonne de norme max (ou aléatoire si tout est nul)
    top = min(50, n)
    top_idx = np.argpartition(x_norm2, -top)[-top:]
    j0 = int(rng.choice(top_idx))

    if x_norm2[j0] <= eps:
        j0 = int(rng.integers(0, n))
    K.append(j0)

    if verbose:
        print(f"[SNPA] t=1 pick={j0} ||x||^2={x_norm2[j0]:.3e}")

    # boucle
    for t in range(2, r + 1):
        # construit W = X[:,K]
        if _is_sparse(Xcsc):
            W = Xcsc[:, K].toarray()
        else:
            W = Xcsc[:, K].copy()

        # normalise les colonnes de W (stabilité numérique)
        norms = np.linalg.norm(W, axis=0) + eps
        W = W / norms[None, :]

        # calcule AtA, AtB
        AtA = W.T @ W  # (t-1 x t-1)
        if _is_sparse(Xcsc):
            AtB = (W.T @ Xcsc).astype(float)  # (t-1 x n) dense
        else:
            AtB = W.T @ Xcsc

        # NNLS: H = argmin_{H>=0} ||X - W H||_F^2  (approx)
        H = nnls_hals(AtA, AtB, n_iter=nnls_iter, eps=eps)

        # résidu colonne par colonne sans former R:
        # ||x||^2 - 2 h^T (W^T x) + h^T (W^T W) h
        cross = np.sum(H * AtB, axis=0)                    # h^T AtB
        quad = np.sum(H * (AtA @ H), axis=0)               # h^T AtA h
        res2 = x_norm2 - 2.0 * cross + quad

        # interdit de re-sélectionner une colonne déjà prise
        res2[np.array(K, dtype=int)] = -np.inf

        top = min(10, n)
        cand = np.argpartition(res2, -top)[-top:]
        j = int(rng.choice(cand))

        K.append(j)
        resid_trace.append(float(res2[j]))

        if verbose:
            print(f"[SNPA] t={t} pick={j} resid^2={res2[j]:.3e}")

    K = np.array(K, dtype=int)

    # W final (non re-normalisé ici, tu peux normaliser ensuite si tu veux)
    if _is_sparse(Xcsc):
        W0 = Xcsc[:, K].toarray()
    else:
        W0 = Xcsc[:, K].copy()

    info = {"resid2_trace": np.array(resid_trace, dtype=float)}
    return K, W0, info
