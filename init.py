import numpy as np

def init_W_random(X: np.ndarray, r: int, seed: int | None = None, nonneg: bool = True):
    """
    Simple random initialization: pick r random columns of X (if nonneg),
    otherwise Gaussian random then nonneg projection if asked.
    """
    rng = np.random.default_rng(seed)
    m, n = X.shape
    if nonneg and np.all(X >= 0):
        idx = rng.choice(n, size=r, replace=False)
        W = X[:, idx].astype(float, copy=True)
        # Avoid zero columns
        norms = np.linalg.norm(W, axis=0)
        norms[norms == 0] = 1.0
        W /= norms[None, :]
        return W
    else:
        W = rng.standard_normal((m, r))
        if nonneg:
            W = np.maximum(0.0, W)
        # normalize columns
        norms = np.linalg.norm(W, axis=0)
        norms[norms == 0] = 1.0
        W /= norms[None, :]
        return W


def warm_start_from_fro_onmf(X: np.ndarray, r: int, iters: int = 3, seed: int | None = None):
    """
    Quick-and-dirty warm start via a few Frobenius ONMF-like steps:
    - Start random W
    - Alternate: assign columns by cosine, compute H (nonneg), update W = X H^T
    Returns W (not necessarily nonnegative).
    """
    rng = np.random.default_rng(seed)
    m, n = X.shape
    W = rng.standard_normal((m, r))
    # normalize columns
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    for _ in range(iters):
        # normalize W
        Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
        A = Wn.T @ X  # r x n
        assign = np.argmax(A, axis=0)
        H = np.zeros((r, n), dtype=float)
        for j in range(n):
            k = assign[j]
            num = float(W[:, k].T @ X[:, j])
            den = float(W[:, k].T @ W[:, k]) + 1e-12
            H[k, j] = max(0.0, num / den)
        # row-normalize H and co-scale W
        row_norms = np.linalg.norm(H, axis=1) + 1e-16
        H = H / row_norms[:, None]
        W = W * row_norms[None, :]
        # update W
        for k in range(r):
            Kj = np.where(H[k, :] > 0)[0]
            if Kj.size > 0:
                W[:, k] = X[:, Kj] @ H[k, Kj].T
        # renormalize
        W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    return W
