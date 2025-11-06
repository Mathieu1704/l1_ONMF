import numpy as np
from l1_onmf import alternating_l1_onmf, L1ONMFOptions

def make_synthetic(m=50, n=300, r=4, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    # ground-truth W (nonneg)
    W = rng.random((m, r))
    # H: one nonzero per column
    H = np.zeros((r, n))
    clusters = rng.integers(0, r, size=n)
    scales = 0.5 + rng.random(n)  # positive scales
    for j in range(n):
        H[clusters[j], j] = scales[j]
    X = W @ H + noise * rng.standard_normal((m, n))
    X = np.maximum(0.0, X)  # keep nonneg for this toy
    return X, W, H

if __name__ == "__main__":
    X, Wgt, Hgt = make_synthetic()
    opts = L1ONMFOptions(r=4, maxiter=50, verbose=True, init="auto")
    W, H, info = alternating_l1_onmf(X, opts)
    print("Done. Final rel-L1 error:", info["rel_l1_errors"][-1])
