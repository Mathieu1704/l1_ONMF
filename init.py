# l1_ONMF/init.py
import numpy as np
try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from .snpa import snpa


def _is_sparse(X) -> bool:
    return sp is not None and sp.isspmatrix(X)


def init_W_snpa(X, r: int, seed: int | None = None):
    # SNPA retourne W0 dense (m x r), ok car r petit
    K, W0, info = snpa(X, r, normalize="l1", nnls_iter=50, seed=seed, verbose=False)
    norms = np.linalg.norm(W0, axis=0)
    norms[norms == 0] = 1.0
    W0 = W0 / norms[None, :]
    return W0, K, info


def init_W_random(X, r: int, seed: int | None = None, nonneg: bool = True):
    """
    Random init (sparse-friendly):
    - Si X est sparse: on échantillonne r colonnes et on les met en dense (m x r)
    - Sinon: on échantillonne colonnes de X ou gaussien
    """
    rng = np.random.default_rng(seed)

    if _is_sparse(X):
        Xcsc = X.tocsc()
        m, n = Xcsc.shape
        idx = rng.choice(n, size=r, replace=False)
        W = Xcsc[:, idx].toarray().astype(float)
        if nonneg:
            W = np.maximum(0.0, W)
    else:
        Xd = np.asarray(X, dtype=float)
        m, n = Xd.shape
        if nonneg and np.all(Xd >= 0):
            idx = rng.choice(n, size=r, replace=False)
            W = Xd[:, idx].copy()
        else:
            W = rng.standard_normal((m, r))
            if nonneg:
                W = np.maximum(0.0, W)

    norms = np.linalg.norm(W, axis=0)
    norms[norms == 0] = 1.0
    W /= norms[None, :]
    return W


def warm_start_from_fro_onmf(X: np.ndarray, r: int, iters: int = 3, seed: int | None = None):
    """
    Warm start Frobenius (dense seulement).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    W = rng.standard_normal((m, r))
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)

    for _ in range(iters):
        Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
        A = Wn.T @ X
        assign = np.argmax(A, axis=0)

        H = np.zeros((r, n), dtype=float)
        for j in range(n):
            k = assign[j]
            num = float(W[:, k].T @ X[:, j])
            den = float(W[:, k].T @ W[:, k]) + 1e-12
            H[k, j] = max(0.0, num / den)

        row_norms = np.linalg.norm(H, axis=1) + 1e-16
        H = H / row_norms[:, None]
        W = W * row_norms[None, :]

        for k in range(r):
            Kj = np.where(H[k, :] > 0)[0]
            if Kj.size > 0:
                W[:, k] = X[:, Kj] @ H[k, Kj].T

        W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)

    return W
