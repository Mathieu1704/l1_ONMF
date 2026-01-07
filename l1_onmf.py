# l1_ONMF/l1_onmf.py
from dataclasses import dataclass
import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from .init import init_W_random, warm_start_from_fro_onmf, init_W_snpa, init_W_kmeans
from .update_h import update_H_l1
from .update_w import update_W_l1
from .normalize import normalize_rows_H_and_rescale_W
from .utils import handle_empty_clusters

@dataclass
class L1ONMFOptions:
    r: int
    maxiter: int = 100
    l1_tol: float = 1e-7
    patience: int = 10
    enforce_W_nonneg: bool = False
    init: str = "auto"
    seed: int | None = None
    n_init: int = 5
    log_errors: bool = True
    verbose: bool = True
    eps: float = 1e-12
    init_prune_top: int | None = None

def _is_sparse(X) -> bool:
    return sp is not None and sp.isspmatrix(X)

def _abs_col_l1(W: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(W), axis=0)

def l1_obj_hard_sparse(Xcsc, W: np.ndarray, H: np.ndarray, eps: float = 1e-12) -> float:
    k, n = H.shape
    assign = np.argmax(H, axis=0).astype(int)
    s = H[assign, np.arange(n)]
    absW_l1 = _abs_col_l1(W)
    
    total = 0.0
    indptr, indices, data = Xcsc.indptr, Xcsc.indices, Xcsc.data

    for j in range(n):
        kk = assign[j]
        sj = float(s[j])
        a, b = indptr[j], indptr[j + 1]
        idx = indices[a:b]
        xnz = data[a:b]

        wk_idx = W[idx, kk]
        total += float(np.sum(np.abs(xnz - sj * wk_idx)))
        total += sj * float(absW_l1[kk] - np.sum(np.abs(wk_idx)))

    return total

def l1_obj_dense(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    R = X - W @ H
    return float(np.sum(np.abs(R)))

def alternating_l1_onmf(X, opts: L1ONMFOptions):
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
            init = "snpa" if (X_is_nonneg and _is_sparse(X)) else "warm_fro"

        if init == "snpa":
            if not X_is_nonneg: raise ValueError("init='snpa' nécessite X >= 0.")
            W0, _, _ = init_W_snpa(Xcsc if _is_sparse(X) else Xd, r, seed=seed)
        elif init == "kmeans":
            W0 = init_W_kmeans(Xcsc if _is_sparse(X) else Xd, r, seed=seed)
        elif init == "random":
            W0 = init_W_random(Xcsc if _is_sparse(X) else Xd, r, seed=seed, nonneg=(opts.enforce_W_nonneg and X_is_nonneg))
        elif init == "warm_fro":
            W0 = warm_start_from_fro_onmf(Xd, r, iters=3, seed=seed)
        else:
            raise ValueError(f"Unknown init='{opts.init}'")

        if opts.enforce_W_nonneg: W0 = np.maximum(0.0, W0)
        col_sums = np.sum(W0, axis=0) + opts.eps
        W0 = W0 / col_sums[None, :]
        return W0

    def _rel_l1_error(W, H) -> float:
        if _is_sparse(X):
            obj = l1_obj_hard_sparse(Xcsc, W, H, eps=opts.eps)
        else:
            obj = l1_obj_dense(Xd, W, H)
        return float(obj / den)

    def _run_once(seed: int | None):
        W = _init_W(seed)
        
        # Init H
        H = update_H_l1(Xcsc if _is_sparse(X) else Xd, W, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
        
        # Gestion Cluster Vide (Méthode Prof)
        assign = np.argmax(H, axis=0)
        W, H, assign = handle_empty_clusters(X, W, H, assign)
        
        row_norms = np.linalg.norm(H, axis=1) + 1e-16
        H = H / row_norms[:, None]

        errs = []
        prev = None
        stall = 0

        if opts.verbose:
            print(f"Start L1-ONMF: m={m} n={n} r={r} init={opts.init} seed={seed}")

        for it in range(1, opts.maxiter + 1):
            # 1. Update H (Hard assignment + optimal scales)
            H = update_H_l1(Xcsc if _is_sparse(X) else Xd, W, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
            
            # 2. Gestion des clusters vides (Split)
            assign = np.argmax(H, axis=0)
            W, H, assign = handle_empty_clusters(X, W, H, assign)
            
            # 3. Normalisation exacte de H et W (Nouveau : respecte strictement WH)
            # Cette ligne garantit que HH^T = I sans changer la valeur de l'erreur
            H, W = normalize_rows_H_and_rescale_W(H, W)

            # 4. Update W (Calcul du candidat via médiane pondérée)
            W_new = update_W_l1(Xcsr if _is_sparse(X) else Xd, H, enforce_W_nonneg=opts.enforce_W_nonneg, eps=opts.eps)
            
            # 5. Stabilisation par inertie (Défendable comme méthode de lissage)
            if it > 1:
                W = 0.7 * W + 0.3 * W_new
            else:
                W = W_new

            # Check erreur
            e = _rel_l1_error(W, H)
            if opts.log_errors: errs.append(e)

            if opts.verbose and it % 10 == 0:
                print(f"  Iter {it:03d} | relL1={e:.6f}")

            if prev is not None and it >= 5:
                rel_impr = (prev - e) / (abs(prev) + 1e-16)
                if rel_impr < opts.l1_tol: stall += 1
                else: stall = 0
                if stall >= opts.patience:
                    break
            prev = e

        info = {"rel_l1_errors": np.array(errs), "num_iter": it, "final_err": float(errs[-1] if errs else 0), "seed": seed}
        return W, H, info

    n_init = max(1, int(opts.n_init))
    base = 0 if opts.seed is None else int(opts.seed)
    best = (None, None, {"final_err": np.inf})
    for t in range(n_init):
        Wt, Ht, infot = _run_once(base + t)
        if infot["final_err"] < best[2]["final_err"]:
            best = (Wt, Ht, infot)
    return best