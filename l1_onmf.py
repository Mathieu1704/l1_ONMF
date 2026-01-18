# l1_ONMF/l1_onmf.py
from dataclasses import dataclass
import warnings
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
    inertia_eta: float = 0.3
    zero_weight: float = 0.05
    track_diagnostics: bool = True


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


def l1_obj_hard_sparse_weighted(Xcsc, W: np.ndarray, H: np.ndarray, zero_weight: float, eps: float = 1e-12) -> float:
    k, n = H.shape
    assign = np.argmax(H, axis=0).astype(int)
    s = H[assign, np.arange(n)]
    absW_l1 = _abs_col_l1(W)

    gamma = float(zero_weight)
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
        total += gamma * sj * float(absW_l1[kk] - np.sum(np.abs(wk_idx)))

    return total


def l1_obj_dense_weighted(X: np.ndarray, W: np.ndarray, H: np.ndarray, zero_weight: float) -> float:
    R = X - (W @ H)
    gamma = float(zero_weight)
    weights = np.ones_like(X, dtype=float)
    weights[X == 0] = gamma
    return float(np.sum(weights * np.abs(R)))


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

        # Fix critique : warm_fro sur X sparse -> fallback (sinon crash)
        if init == "warm_fro" and _is_sparse(X):
            if opts.verbose:
                print("WARN: init='warm_fro' n'est pas supporté pour X sparse -> fallback vers init='random'.")
            init = "random"

        if init == "snpa":
            if not X_is_nonneg:
                raise ValueError("init='snpa' nécessite X >= 0.")
            W0, _, _ = init_W_snpa(Xcsc if _is_sparse(X) else Xd, r, seed=seed)

        elif init == "kmeans":
            W0 = init_W_kmeans(Xcsc if _is_sparse(X) else Xd, r, seed=seed)

        elif init == "random":
            W0 = init_W_random(
                Xcsc if _is_sparse(X) else Xd,
                r,
                seed=seed,
                nonneg=(opts.enforce_W_nonneg and X_is_nonneg),
            )

        elif init == "warm_fro":
            # ici forcément dense
            W0 = warm_start_from_fro_onmf(Xd, r, iters=3, seed=seed)

        else:
            raise ValueError(f"Unknown init='{opts.init}'")

        if opts.enforce_W_nonneg:
            W0 = np.maximum(0.0, W0)

        # Normalisation robuste (marche aussi si W contient des négatifs)
        col_l1 = np.sum(np.abs(W0), axis=0) + opts.eps
        W0 = W0 / col_l1[None, :]
        return W0

    def _rel_l1_error_pure(W, H) -> float:
        if _is_sparse(X):
            obj = l1_obj_hard_sparse(Xcsc, W, H, eps=opts.eps)
        else:
            obj = l1_obj_dense(Xd, W, H)
        return float(obj / den)

    def _rel_l1_error_weighted(W, H) -> float:
        if _is_sparse(X):
            obj = l1_obj_hard_sparse_weighted(Xcsc, W, H, zero_weight=opts.zero_weight, eps=opts.eps)
        else:
            obj = l1_obj_dense_weighted(Xd, W, H, zero_weight=opts.zero_weight)
        return float(obj / den)

    def _run_once(seed: int | None):
        W = _init_W(seed)

        warned_negW = False

        def _guard_negative_W(Wmat: np.ndarray):
            nonlocal warned_negW
            if warned_negW:
                return
            if (not opts.enforce_W_nonneg) and np.any(Wmat < -1e-12):
                warnings.warn(
                    "W contient des valeurs négatives alors que enforce_W_nonneg=False. "
                    "Les updates H/W basées sur la médiane pondérée sont surtout justifiées pour le cas NMF (W >= 0). "
                    "Le code continue, mais les garanties de correction/convergence peuvent être perdues. "
                    "Si tu veux un comportement théoriquement propre, active enforce_W_nonneg=True.",
                    RuntimeWarning,
                )
                warned_negW = True

        # Init H (cohérent avec gamma)
        _guard_negative_W(W)
        H = update_H_l1(
            Xcsc if _is_sparse(X) else Xd,
            W,
            enforce_W_nonneg=opts.enforce_W_nonneg,
            eps=opts.eps,
            zero_weight=opts.zero_weight,
        )

        assign = np.argmax(H, axis=0)
        W, H, assign = handle_empty_clusters(X, W, H, assign)
        H, W = normalize_rows_H_and_rescale_W(H, W)

        errs_pure = []
        errs_weighted = []
        prev = None
        stall = 0

        if opts.verbose:
            print(f"Start L1-ONMF: m={m} n={n} r={r} init={opts.init} seed={seed}")

        w_changes = []
        w_nnz_frac = []
        min_cluster_sizes = []

        for it in range(1, opts.maxiter + 1):
            # 1) Update H
            _guard_negative_W(W)
            H = update_H_l1(
                Xcsc if _is_sparse(X) else Xd,
                W,
                enforce_W_nonneg=opts.enforce_W_nonneg,
                eps=opts.eps,
                zero_weight=opts.zero_weight,
            )

            # 2) Clusters vides
            assign = np.argmax(H, axis=0)
            W, H, assign = handle_empty_clusters(X, W, H, assign)

            # 3) Normalisation exacte (préserve WH)
            H, W = normalize_rows_H_and_rescale_W(H, W)

            # 4) Update W + inertie
            W_prev = W.copy()

            _guard_negative_W(W)
            W_new = update_W_l1(
                Xcsr if _is_sparse(X) else Xd,
                H,
                enforce_W_nonneg=opts.enforce_W_nonneg,
                eps=opts.eps,
                zero_weight=opts.zero_weight,
            )

            eta = float(opts.inertia_eta)
            if eta < 0.0:
                eta = 0.0
            if eta > 1.0:
                eta = 1.0

            W = (1.0 - eta) * W + eta * W_new

            # Diagnostics
            if opts.track_diagnostics:
                num = float(np.linalg.norm(W - W_prev))
                denW = float(np.linalg.norm(W_prev) + 1e-16)
                w_changes.append(num / denW)

                thr = 1e-12
                w_nnz_frac.append(float(np.mean(np.abs(W) > thr)))

                sizes = np.bincount(assign, minlength=r)
                min_cluster_sizes.append(int(sizes.min()))

            # Erreurs
            e_pure = _rel_l1_error_pure(W, H)
            e_w = _rel_l1_error_weighted(W, H)

            if opts.log_errors:
                errs_pure.append(e_pure)
                errs_weighted.append(e_w)

            if opts.verbose and it % 10 == 0:
                print(f"  Iter {it:03d} | relL1(pure)={e_pure:.6f} | relL1(weighted)={e_w:.6f}")

            # Early stop sur l'objectif optimisé
            e_for_stop = e_w if abs(float(opts.zero_weight) - 1.0) > 1e-15 else e_pure

            if prev is not None and it >= 5:
                rel_impr = (prev - e_for_stop) / (abs(prev) + 1e-16)
                if rel_impr < opts.l1_tol:
                    stall += 1
                else:
                    stall = 0
                if stall >= opts.patience:
                    break
            prev = e_for_stop

        info = {
            "rel_l1_errors": np.array(errs_pure, dtype=float),
            "rel_l1_errors_weighted": np.array(errs_weighted, dtype=float),
            "num_iter": it,
            "final_err": float(errs_pure[-1] if errs_pure else 0.0),
            "final_err_weighted": float(errs_weighted[-1] if errs_weighted else 0.0),
            "seed": seed,
        }

        if opts.track_diagnostics:
            info["w_changes"] = np.array(w_changes, dtype=float)
            info["w_nnz_frac"] = np.array(w_nnz_frac, dtype=float)
            info["min_cluster_sizes"] = np.array(min_cluster_sizes, dtype=int)

        return W, H, info

    n_init = max(1, int(opts.n_init))
    base = 0 if opts.seed is None else int(opts.seed)

    use_weighted_selection = (abs(float(opts.zero_weight) - 1.0) > 1e-15)

    best_W = None
    best_H = None
    best_info = None
    best_score = np.inf

    for t in range(n_init):
        Wt, Ht, infot = _run_once(base + t)
        current_score = infot["final_err_weighted"] if use_weighted_selection else infot["final_err"]

        if current_score < best_score:
            best_score = current_score
            best_W = Wt
            best_H = Ht
            best_info = infot

    return best_W, best_H, best_info
