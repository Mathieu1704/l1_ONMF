from dataclasses import dataclass
import numpy as np

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



@dataclass
class L1ONMFOptions:
    r: int
    maxiter: int = 100
    delta: float = 1e-6           # stopping on ||H - H_prev||_F
    enforce_W_nonneg: bool = False
    init: str = "warm_fro"            # "auto" | "random" | "warm_fro"
    seed: int | None = None
    log_errors: bool = True
    verbose: bool = True
    eps: float = 1e-12
    n_init: int = 10


def rel_l1_error(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    num = np.sum(np.abs(X - W @ H))
    den = np.sum(np.abs(X)) + 1e-16
    return float(num / den)



def alternating_l1_onmf(X: np.ndarray, opts: L1ONMFOptions):
    """
    Main loop for L1-ONMF with hard clustering induced by H>=0, H H^T = I.
    Multi-start via opts.n_init (keep best final L1 error).
    """
    # (pour l’instant) densifie si sparse
    if sp is not None and sp.isspmatrix(X):
        X = X.toarray()
    else:
        X = np.asarray(X, dtype=float)

    m, n = X.shape
    r = opts.r

    def _init_W(seed: int | None):
        if opts.init == "random":
            W0 = init_W_random(X, r, seed=seed,
                               nonneg=opts.enforce_W_nonneg and np.all(X >= 0))
        elif opts.init == "warm_fro":
            W0 = warm_start_from_fro_onmf(X, r, iters=3, seed=seed)
            if opts.enforce_W_nonneg:
                W0 = np.maximum(0.0, W0)
        else:  # auto
            if np.all(X >= 0):
                W0 = init_W_random(X, r, seed=seed, nonneg=True)
            else:
                W0 = warm_start_from_fro_onmf(X, r, iters=3, seed=seed)
                if opts.enforce_W_nonneg:
                    W0 = np.maximum(0.0, W0)
        return W0

    def _run_once(seed: int | None):
        W = _init_W(seed)

        # init H propre (évite un départ H=0 + clusters vides)
        H = update_H_l1(X, W, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
        H = ensure_nonempty_clusters(H)
        H, W = normalize_rows_H_and_rescale_W(H, W)

        errs = []
        if opts.verbose:
            print(f"Starting L1-ONMF: m={m}, n={n}, r={r}, maxiter={opts.maxiter}, seed={seed}")

        for it in range(1, opts.maxiter + 1):
            H_prev = H.copy()

            if opts.verbose:
                err_before = rel_l1_error(X, W, H)
                print(f"\nIter {it:03d} | début : rel L1 err = {err_before:.6f}")

            # --- Update H ---
            H = update_H_l1(X, W, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
            H = ensure_nonempty_clusters(H)
            H, W = normalize_rows_H_and_rescale_W(H, W)

            if opts.verbose:
                err_after_H = rel_l1_error(X, W, H)
                print(f"Iter {it:03d} | après update_H+norm : rel L1 err = {err_after_H:.6f}")

            # --- Update W ---
            W = update_W_l1(X, H, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
            err_after_W = rel_l1_error(X, W, H)

            if opts.verbose:
                print(f"Iter {it:03d} | après update_W : rel L1 err = {err_after_W:.6f}")

            if opts.log_errors:
                errs.append(err_after_W)

            # stopping sur H
            diff = np.linalg.norm(H - H_prev, ord="fro")
            if diff < opts.delta and it >= 3:
                if opts.verbose:
                    print(f"Converged at iter {it} (||H-H_prev||_F={diff:.3e}).")
                break

        final_err = errs[-1] if len(errs) else rel_l1_error(X, W, H)
        info = {"rel_l1_errors": np.array(errs), "num_iter": it, "final_err": float(final_err), "seed": seed}
        return W, H, info

    # ---- Multi-start ----
    n_init = max(1, int(getattr(opts, "n_init", 1)))
    base_seed = 0 if opts.seed is None else int(opts.seed)

    best_W, best_H, best_info = None, None, None
    best_err = np.inf

    for t in range(n_init):
        seed_t = base_seed + t
        Wt, Ht, infot = _run_once(seed_t)

        if infot["final_err"] < best_err:
            best_err = infot["final_err"]
            best_W, best_H, best_info = Wt, Ht, infot

    return best_W, best_H, best_info

