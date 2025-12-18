# l1_ONMF/l1_onmf.py
from dataclasses import dataclass
import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from .init import init_W_random, warm_start_from_fro_onmf, init_W_snpa
from .update_h import update_H_l1
from .update_w import update_W_l1
from .normalize import normalize_rows_H_and_rescale_W
from .utils import ensure_nonempty_clusters


@dataclass
class L1ONMFOptions:
    r: int
    maxiter: int = 100
    l1_tol: float = 1e-7
    patience: int = 5
    enforce_W_nonneg: bool = False
    init: str = "auto"           # "auto" | "random" | "warm_fro" | "snpa"
    seed: int | None = None
    n_init: int = 5
    log_errors: bool = True
    verbose: bool = True
    eps: float = 1e-12
    # Option INIT uniquement (hors énoncé, donc uniquement pour démarrer mieux)
    init_prune_top: int | None = None   # ex: 500 pour docs (mettre None pour désactiver)


def _is_sparse(X) -> bool:
    return sp is not None and sp.isspmatrix(X)


def _abs_col_l1(W: np.ndarray) -> np.ndarray:
    # ||w_k||_1 pour chaque colonne k (valable même si W signé)
    return np.sum(np.abs(W), axis=0)


def l1_obj_hard_sparse(Xcsc, W: np.ndarray, H: np.ndarray, eps: float = 1e-12) -> float:
    """
    Objectif exact ||X - WH||_1 pour X sparse, en exploitant hard clustering de H.
    Ne forme jamais WH.
    """
    k, n = H.shape
    assign = np.argmax(H, axis=0).astype(int)
    s = H[assign, np.arange(n)]

    absW_l1 = _abs_col_l1(W)  # (k,)

    total = 0.0
    indptr, indices, data = Xcsc.indptr, Xcsc.indices, Xcsc.data

    for j in range(n):
        kk = assign[j]
        sj = float(s[j])
        a, b = indptr[j], indptr[j + 1]
        idx = indices[a:b]
        xnz = data[a:b]

        wk_idx = W[idx, kk]
        # coût sur nnz : sum |x - s*w|
        total += float(np.sum(np.abs(xnz - sj * wk_idx)))

        # coût sur zéros : sum_{d not in idx} |0 - s*w_d| = s * (||w||_1 - sum_{idx} |w|)
        total += sj * float(absW_l1[kk] - np.sum(np.abs(wk_idx)))

    return total


def l1_obj_dense(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    R = X - W @ H
    return float(np.sum(np.abs(R)))


def alternating_l1_onmf(X, opts: L1ONMFOptions):
    # --- Prépare X sans densifier inutilement ---
    if _is_sparse(X):
        Xcsc = X.tocsc().astype(float)
        Xcsr = X.tocsr().astype(float)
        m, n = Xcsc.shape
        den = float(np.sum(np.abs(Xcsc.data)) + 1e-16)
        X_is_nonneg = bool((Xcsc.data >= 0).all())
    else:
        Xd = np.asarray(X, dtype=float)
        m, n = Xd.shape
        den = float(np.sum(np.abs(Xd)) + 1e-16)
        X_is_nonneg = bool(np.all(Xd >= 0))

    r = int(opts.r)

    def _init_W(seed: int | None):
        init = opts.init

        if init == "auto":
            if X_is_nonneg:
                init = "snpa" if _is_sparse(X) else "random"
            else:
                init = "warm_fro"

        if init == "snpa":
            if not X_is_nonneg:
                raise ValueError("init='snpa' nécessite X >= 0.")
            W0, _, _ = init_W_snpa(Xcsc if _is_sparse(X) else Xd, r, seed=seed)
        elif init == "random":
            W0 = init_W_random(Xcsc if _is_sparse(X) else Xd, r, seed=seed,
                               nonneg=(opts.enforce_W_nonneg and X_is_nonneg))
        elif init == "warm_fro":
            if _is_sparse(X):
                # Évite de densifier un énorme doc-term matrix
                W0 = init_W_random(Xcsc, r, seed=seed, nonneg=(opts.enforce_W_nonneg and X_is_nonneg))
            else:
                W0 = warm_start_from_fro_onmf(Xd, r, iters=3, seed=seed)
        else:
            raise ValueError(f"Unknown init='{opts.init}'")

        if opts.enforce_W_nonneg:
            W0 = np.maximum(0.0, W0)

        # Pruning INIT uniquement (optionnel)
        if opts.init_prune_top is not None and opts.init_prune_top > 0 and opts.init_prune_top < W0.shape[0]:
            top = int(opts.init_prune_top)
            for kk in range(W0.shape[1]):
                col = W0[:, kk]
                if np.count_nonzero(col) > top:
                    keep = np.argpartition(col, -top)[-top:]
                    mask = np.zeros_like(col, dtype=bool)
                    mask[keep] = True
                    col[~mask] = 0.0
                    W0[:, kk] = col

        # normalise colonnes
        norms = np.linalg.norm(W0, axis=0) + opts.eps
        return W0 / norms[None, :]

    def _rel_l1_error(W, H) -> float:
        if _is_sparse(X):
            obj = l1_obj_hard_sparse(Xcsc, W, H, eps=opts.eps)
        else:
            obj = l1_obj_dense(Xd, W, H)
        return float(obj / den)

    def _run_once(seed: int | None):
        W = _init_W(seed)

        # init H via L1 assignment exact (sparse-friendly)
        H = update_H_l1(Xcsc if _is_sparse(X) else Xd, W,
                        enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
        assign = np.argmax(H, axis=0)
        print("  #clusters sizes:", np.bincount(assign, minlength=opts.r))
        print("  H nonzeros:", np.count_nonzero(H))

        H = ensure_nonempty_clusters(H)
        H, W = normalize_rows_H_and_rescale_W(H, W)

        errs = []
        prev = None
        stall = 0

        if opts.verbose:
            print(f"Start L1-ONMF (exact): m={m} n={n} r={r} init={opts.init} seed={seed}")

        for it in range(1, opts.maxiter + 1):
            if opts.verbose:
                print(f"\nIter {it:03d} | start relL1={_rel_l1_error(W,H):.6f}")

            # H update
            H = update_H_l1(Xcsc if _is_sparse(X) else Xd, W,
                            enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
            assign = np.argmax(H, axis=0)
            print("  #clusters sizes:", np.bincount(assign, minlength=opts.r))
            print("  H nonzeros:", np.count_nonzero(H))

            H = ensure_nonempty_clusters(H)
            H, W = normalize_rows_H_and_rescale_W(H, W)

            # W update
            W = update_W_l1(Xcsr if _is_sparse(X) else Xd, H,
                            enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
            print("  W nnz:", np.count_nonzero(W), " / ", W.size)


            e = _rel_l1_error(W, H)
            if opts.log_errors:
                errs.append(e)
            if opts.verbose:
                print(f"Iter {it:03d} | end   relL1={e:.6f}")

            if prev is not None and it >= 3:
                rel_impr = (prev - e) / (abs(prev) + 1e-16)
                stall = stall + 1 if rel_impr < opts.l1_tol else 0
                if stall >= opts.patience:
                    if opts.verbose:
                        print(f"Stop (stagnation): rel_impr={rel_impr:.3e} < {opts.l1_tol}")
                    break
            prev = e

        info = {
            "rel_l1_errors": np.array(errs, dtype=float),
            "num_iter": it,
            "final_err": float(errs[-1] if errs else _rel_l1_error(W,H)),
            "seed": seed,
        }
        return W, H, info

    # Multi-start
    n_init = max(1, int(opts.n_init))
    base = 0 if opts.seed is None else int(opts.seed)

    best = (None, None, {"final_err": np.inf})
    for t in range(n_init):
        Wt, Ht, infot = _run_once(base + t)
        if infot["final_err"] < best[2]["final_err"]:
            best = (Wt, Ht, infot)

    return best
