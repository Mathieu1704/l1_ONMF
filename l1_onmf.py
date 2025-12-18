from dataclasses import dataclass
import numpy as np

<<<<<<< Updated upstream
from .init import init_W_random, warm_start_from_fro_onmf
from .update_h import update_H_l1
from .update_w import update_W_l1
from .normalize import normalize_rows_H_and_rescale_W
from .metrics import rel_l1_error
from .utils import ensure_nonempty_clusters
=======
try:
    import scipy.sparse as sp
except ImportError:
    sp = None

try:
    from .init import init_W_random, warm_start_from_fro_onmf
    from .update_h import update_H_l1
    from .update_w import update_W_l1
    from .normalize import normalize_rows_H_and_rescale_W
    from .utils import ensure_nonempty_clusters
except ImportError:
    from init import init_W_random, warm_start_from_fro_onmf
    from update_h import update_H_l1
    from update_w import update_W_l1
    from normalize import normalize_rows_H_and_rescale_W
    from utils import ensure_nonempty_clusters

>>>>>>> Stashed changes

@dataclass
class L1ONMFOptions:
    r: int
    maxiter: int = 100
    delta: float = 1e-6
    enforce_W_nonneg: bool = True
<<<<<<< Updated upstream
    init: str = "auto"            # "auto" | "random" | "warm_fro"
=======
    init: str = "warm_fro"
>>>>>>> Stashed changes
    seed: int | None = None
    log_errors: bool = False     # sur gros sparse, laisse False
    verbose: bool = False
    eps: float = 1e-12


<<<<<<< Updated upstream
def alternating_l1_onmf(X: np.ndarray, opts: L1ONMFOptions):
=======
def rel_l1_error(X, W: np.ndarray, H: np.ndarray) -> float:
>>>>>>> Stashed changes
    """
    Attention: coûteux sur gros sparse. À laisser désactivé en prod docs.
    """
<<<<<<< Updated upstream
    X = np.asarray(X, dtype=float)
    m, n = X.shape
=======
    WH = W @ H
    if sp is not None and sp.isspmatrix(X):
        # Calcul exact mais coûteux: densifie WH (déjà dense) et compare à X densifié -> à éviter sur classic
        Xd = X.toarray()
        num = np.sum(np.abs(Xd - WH))
        den = np.sum(np.abs(Xd)) + 1e-16
        return float(num / den)
    else:
        X = np.asarray(X, dtype=float)
        num = np.sum(np.abs(X - WH))
        den = np.sum(np.abs(X)) + 1e-16
        return float(num / den)


def alternating_l1_onmf(X, opts: L1ONMFOptions):
    """
    L1-ONMF with hard clustering induced by H>=0, H H^T = I.
    """
    is_sparse = (sp is not None and sp.isspmatrix(X))

    if is_sparse:
        # Important: CSC pour update_H (accès colonnes), CSR pour update_W (accès lignes)
        X_csc = X.tocsc()
        X_csr = X.tocsr()
        m, n = X_csc.shape
    else:
        X = np.asarray(X, dtype=float)
        X_csc = None
        X_csr = None
        m, n = X.shape

>>>>>>> Stashed changes
    r = opts.r

    # --- Init W ---
    if opts.init == "random":
        X_for_init = X_csc if is_sparse else X
        W = init_W_random(np.asarray(X_for_init.toarray() if is_sparse else X_for_init), r,
                          seed=opts.seed,
                          nonneg=opts.enforce_W_nonneg)
    elif opts.init == "warm_fro":
        X_for_init = X_csc if is_sparse else X
        W = warm_start_from_fro_onmf(np.asarray(X_for_init.toarray() if is_sparse else X_for_init),
                                     r, iters=3, seed=opts.seed)
        if opts.enforce_W_nonneg:
            W = np.maximum(0.0, W)
    else:  # auto
        X_for_init = X_csc if is_sparse else X
        W = init_W_random(np.asarray(X_for_init.toarray() if is_sparse else X_for_init), r,
                          seed=opts.seed,
                          nonneg=True)

    H = np.zeros((r, n), dtype=float)
    errs: list[float] = []
    H_prev = H.copy()

    if opts.verbose:
        print(f"Starting L1-ONMF: m={m}, n={n}, r={r}, maxiter={opts.maxiter}")

<<<<<<< Updated upstream
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
=======
    it = 0
    for it in range(1, opts.maxiter + 1):
        H_prev = H.copy()

        # --- Update H (sparse CSC conseillé) ---
        if is_sparse:
            H = update_H_l1(X_csc, W, eps=opts.eps)
        else:
            H = update_H_l1(X, W, eps=opts.eps)

        H = ensure_nonempty_clusters(H, X_csc if is_sparse else X)

        # --- Normalize rows of H and rescale W ---
        H, W = normalize_rows_H_and_rescale_W(H, W)

        # --- Update W (sparse CSR conseillé) ---
        if is_sparse:
            W = update_W_l1(X_csr, H, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
        else:
            W = update_W_l1(X, H, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)

        if opts.log_errors or opts.verbose:
            err = rel_l1_error(X_csc if is_sparse else X, W, H)
            if opts.log_errors:
                errs.append(err)
            if opts.verbose:
                print(f"Iter {it:03d} | rel L1 err = {err:.6f}")

>>>>>>> Stashed changes
        diff = np.linalg.norm(H - H_prev, ord="fro")
        if diff < opts.delta and it >= 3:
            if opts.verbose:
                print(f"Converged at iter {it} (||H-H_prev||_F={diff:.3e}).")
            break

    return W, H, {"rel_l1_errors": np.array(errs), "num_iter": it}
