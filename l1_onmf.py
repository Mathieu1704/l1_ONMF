from dataclasses import dataclass
import numpy as np

# Imports robustes : d'abord relatifs (package), sinon plats (modules locaux)
try:
    from .init import init_W_random, warm_start_from_fro_onmf
    from .update_h import update_H_l1
    from .update_w import update_W_l1
    from .normalize import normalize_rows_H_and_rescale_W
    from .metrics import rel_l1_error
    from .utils import ensure_nonempty_clusters
except ImportError:
    from init import init_W_random, warm_start_from_fro_onmf
    from update_h import update_H_l1
    from update_w import update_W_l1
    from normalize import normalize_rows_H_and_rescale_W
    from metrics import rel_l1_error
    from utils import ensure_nonempty_clusters



@dataclass
class L1ONMFOptions:
    r: int
    maxiter: int = 100
    delta: float = 1e-6           # stopping on ||H - H_prev||_F
    enforce_W_nonneg: bool = True
    init: str = "auto"            # "auto" | "random" | "warm_fro"
    seed: int | None = None
    log_errors: bool = True
    verbose: bool = True
    eps: float = 1e-12

def rel_l1_error(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    num = np.sum(np.abs(X - W @ H))
    den = np.sum(np.abs(X)) + 1e-16
    return float(num / den)



def alternating_l1_onmf(X: np.ndarray, opts: L1ONMFOptions):
    """
    Main 2-BCD loop for L1-ONMF with hard clustering induced by H>=0, H H^T = I.
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    r = opts.r

    # --- Init W ---
    if opts.init == "random":
        W = init_W_random(X, r, seed=opts.seed, nonneg=opts.enforce_W_nonneg and np.all(X >= 0))
    elif opts.init == "warm_fro":
        W = warm_start_from_fro_onmf(X, r, iters=3, seed=opts.seed)
        if opts.enforce_W_nonneg:
            W = np.maximum(0.0, W)
    else:  # auto
        if np.all(X >= 0):
            W = init_W_random(X, r, seed=opts.seed, nonneg=True)
        else:
            W = warm_start_from_fro_onmf(X, r, iters=3, seed=opts.seed)
            if opts.enforce_W_nonneg:
                W = np.maximum(0.0, W)

    H = np.zeros((r, n), dtype=float)
    errs = []
    H_prev = H.copy()

    if opts.verbose:
        print("Starting L1-ONMF: m={}, n={}, r={}, maxiter={}".format(m, n, r, opts.maxiter))

    for it in range(1, opts.maxiter + 1):
        # --- Update H (assignments + scales) ---
        H_prev = H
        H = update_H_l1(X, W, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)

        # Safety: avoid empty clusters (rare)
        H = ensure_nonempty_clusters(H, X)

        # --- Normalize rows of H and co-scale W (preserves WH) ---
        H, W = normalize_rows_H_and_rescale_W(H, W)

        # --- Update W (coordinate-wise weighted medians) ---
        W = update_W_l1(X, H, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)

        # --- Error / stopping ---
        if opts.log_errors:
            err = rel_l1_error(X, W, H)
            errs.append(err)
            if opts.verbose:
                print(f"Iter {it:03d} | rel L1 err = {err:.6f}")

        # stopping on H change
        diff = np.linalg.norm(H - H_prev, ord="fro")
        if diff < opts.delta and it >= 3:
            if opts.verbose:
                print(f"Converged at iter {it} (||H-H_prev||_F={diff:.3e}).")
            break

    return W, H, {"rel_l1_errors": np.array(errs), "num_iter": it}
