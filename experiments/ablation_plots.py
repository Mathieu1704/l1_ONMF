# experiments/ablation_plots.py
import sys
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

# --- SETUP PATH ---
PKG_PARENT = Path(__file__).resolve().parents[2]
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from l1_ONMF.datasets import load_doc_mat, load_hsi_mat
from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
from l1_ONMF.metrics import clustering_accuracy_hungarian


# -----------------------------
# Dirs
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "data" / "docs"
HSI_DIR = ROOT / "data" / "hsi"

OUT_DIR = Path(__file__).resolve().parent / "plots" / "ablations_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAX_FEATURES = 5000


# -----------------------------
# Ablation configs
# -----------------------------
CONFIGS = [
    ("PURE (γ=1, η=1)", 1.0, 1.0),
    ("Only γ (γ=0.05, η=1)", 0.05, 1.0),
    ("Only inertia (γ=1, η=0.3)", 1.0, 0.3),
    ("γ + inertia (γ=0.05, η=0.3)", 0.05, 0.3),
]


# -----------------------------
# Preprocess
# -----------------------------
def preprocess_docs(X_raw, max_features=DEFAULT_MAX_FEATURES):
    # X_raw: (words x docs) souvent sparse
    X_t = X_raw.T  # docs x words
    tfidf = TfidfTransformer(norm="l1", use_idf=True, smooth_idf=True)
    X_t = tfidf.fit_transform(X_t)

    # reduce vocab
    if max_features is not None and X_t.shape[1] > int(max_features):
        word_scores = np.asarray(X_t.sum(axis=0)).ravel()
        idx = np.argsort(word_scores)[-int(max_features) :]
        X_t = X_t[:, idx]

    X = X_t.T  # words x docs
    X = normalize(X, norm="l1", axis=0)
    return X


def preprocess_hsi(X):
    # HSI est en général dense & nonneg
    X = np.asarray(X, dtype=float)
    # normalise colonnes (pixels) pour être cohérent avec ONMF hard-clustering
    X = normalize(X, norm="l1", axis=0)
    return X


# -----------------------------
# Helpers I/O
# -----------------------------
def list_mat_files(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mat"])


def safe_accuracy(y_true, H):
    if y_true is None:
        return None
    y_pred = np.argmax(H, axis=0) + 1
    return float(clustering_accuracy_hungarian(y_true, y_pred))


def extract_final_metrics(info: dict):
    """
    On supporte plusieurs variantes de ton l1_onmf.py (suivant ce que tu as déjà modifié).
    """
    # pure
    if "final_pure" in info:
        final_pure = float(info["final_pure"])
    elif "final_err" in info:
        final_pure = float(info["final_err"])
    else:
        rel = info.get("rel_l1_errors", None)
        final_pure = float(rel[-1]) if rel is not None and len(rel) else np.nan

    # weighted
    if "final_weighted" in info:
        final_weighted = float(info["final_err_weighted"])
    else:
        # fallback: si pas loggué, on met pure (pour ne pas casser le CSV)
        final_weighted = final_pure

    num_iter = int(info.get("num_iter", 0))
    return final_pure, final_weighted, num_iter


# -----------------------------
# Core runs
# -----------------------------
def run_single(X, y_true, r, gamma, eta, seed, maxiter):
    opts = L1ONMFOptions(
        r=int(r),
        maxiter=int(maxiter),
        init="kmeans",
        n_init=1,
        seed=int(seed),
        verbose=False,
        log_errors=True,
        enforce_W_nonneg=True,
        zero_weight=float(gamma),
        inertia_eta=float(eta),
        track_diagnostics=True,
        patience=int(maxiter) + 1,  # pour garder longueur comparable
    )
    W, H, info = alternating_l1_onmf(X, opts)
    acc = safe_accuracy(y_true, H)
    final_pure, final_weighted, num_iter = extract_final_metrics(info)
    return {
        "acc": acc,
        "final_pure": final_pure,
        "final_weighted": final_weighted,
        "num_iter": num_iter,
        "info": info,
    }


def summarize_many(runs):
    def _mean_std(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if len(vals) == 0:
            return (None, None)
        arr = np.asarray(vals, dtype=float)
        if arr.size == 1:
            return (float(arr[0]), 0.0)
        return (float(arr.mean()), float(arr.std(ddof=1)))

    acc_mean, acc_std = _mean_std([r["acc"] for r in runs])
    pure_mean, pure_std = _mean_std([r["final_pure"] for r in runs])
    w_mean, w_std = _mean_std([r["final_weighted"] for r in runs])
    it_mean, it_std = _mean_std([r["num_iter"] for r in runs])

    return {
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "pure_mean": pure_mean,
        "pure_std": pure_std,
        "weighted_mean": w_mean,
        "weighted_std": w_std,
        "iter_mean": it_mean,
        "iter_std": it_std,
    }


def ablation_one_dataset(kind, dataset_path, seeds, maxiter, max_features, make_plots=False):
    name = dataset_path.name

    # ---- load + preprocess
    if kind == "docs":
        X_raw, y_true, r = load_doc_mat(str(dataset_path))
        X = preprocess_docs(X_raw, max_features=max_features)
    elif kind == "hsi":
        out = load_hsi_mat(str(dataset_path))

        # On essaye d’être robuste aux signatures possibles.
        # Cas attendu: (X, y_true, r) OU (X, y_true, r, ...) OU (X, r, ...)
        X = out[0]
        y_true = None
        r = None

        if len(out) >= 3:
            # heuristique: si out[1] ressemble à un vecteur de labels, on le prend
            if isinstance(out[1], np.ndarray) and out[1].ndim == 1:
                y_true = out[1]
                r = out[2]
            else:
                # sinon on suppose que out[1] est r
                r = out[1]
        elif len(out) == 2:
            r = out[1]

        if r is None:
            raise RuntimeError(f"Impossible d'inférer r depuis load_hsi_mat({dataset_path}).")

        X = preprocess_hsi(X)
    else:
        raise ValueError("kind must be 'docs' or 'hsi'")

    # ---- run configs
    per_config = []
    for cfg_name, gamma, eta in CONFIGS:
        runs = []
        for sd in seeds:
            runs.append(run_single(X, y_true, r, gamma, eta, seed=sd, maxiter=maxiter))
        summ = summarize_many(runs)
        per_config.append({
            "dataset": name,
            "kind": kind,
            "config": cfg_name,
            "gamma": gamma,
            "eta": eta,
            **summ,
        })

    # ---- deltas vs PURE (sur les moyennes)
    # baseline = première config (PURE)
    base = per_config[0]
    for row in per_config:
        if row["pure_mean"] is not None and base["pure_mean"] is not None:
            row["delta_pure_vs_pure"] = row["pure_mean"] - base["pure_mean"]
        else:
            row["delta_pure_vs_pure"] = None

        if row["weighted_mean"] is not None and base["weighted_mean"] is not None:
            row["delta_weighted_vs_pure"] = row["weighted_mean"] - base["weighted_mean"]
        else:
            row["delta_weighted_vs_pure"] = None

        if row["acc_mean"] is not None and base["acc_mean"] is not None:
            row["delta_acc_vs_pure"] = row["acc_mean"] - base["acc_mean"]
        else:
            row["delta_acc_vs_pure"] = None

    # ---- optional plots (1 seed = seeds[0]) pour limiter le temps
    if make_plots and len(seeds) > 0:
        sd0 = seeds[0]
        curves = []
        for cfg_name, gamma, eta in CONFIGS:
            one = run_single(X, y_true, r, gamma, eta, seed=sd0, maxiter=maxiter)
            info = one["info"]
            curves.append((cfg_name, info.get("rel_l1_errors", None)))

        # objective curves (rel_l1_errors)
        plt.figure()
        for cfg_name, y in curves:
            if y is None or len(y) == 0:
                continue
            plt.plot(np.arange(1, len(y) + 1), y, label=cfg_name)
        plt.xlabel("Iteration")
        plt.ylabel("Relative L1 objective")
        plt.grid(True)
        plt.title(f"{kind}:{name} objective (seed={sd0})")
        plt.legend()
        plt.tight_layout()
        outp = OUT_DIR / f"{kind}_{name}_obj_seed{sd0}.png"
        plt.savefig(outp, dpi=150)
        plt.close()

    return per_config


def write_csv(rows, outpath: Path):
    if len(rows) == 0:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # stable column order
    fieldnames = [
        "kind", "dataset", "config", "gamma", "eta",
        "acc_mean", "acc_std",
        "pure_mean", "pure_std",
        "weighted_mean", "weighted_std",
        "iter_mean", "iter_std",
        "delta_acc_vs_pure",
        "delta_pure_vs_pure",
        "delta_weighted_vs_pure",
    ]

    with outpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["docs", "hsi", "all"], default="all")
    ap.add_argument("--maxiter", type=int, default=50)
    ap.add_argument("--max_features", type=int, default=DEFAULT_MAX_FEATURES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--plots", action="store_true", help="save 1 objective plot per dataset (seed=seeds[0])")
    args = ap.parse_args()

    all_rows = []

    if args.mode in ("docs", "all"):
        doc_files = list_mat_files(DOC_DIR)
        print(f"[DOCS] found {len(doc_files)} .mat files in {DOC_DIR}")
        for p in doc_files:
            try:
                rows = ablation_one_dataset(
                    kind="docs",
                    dataset_path=p,
                    seeds=args.seeds,
                    maxiter=args.maxiter,
                    max_features=args.max_features,
                    make_plots=args.plots,
                )
                all_rows.extend(rows)
                # mini print résumé
                best = min(rows, key=lambda r: r["weighted_mean"] if r["weighted_mean"] is not None else np.inf)
                print(f"  - {p.name}: best(weighted_mean) = {best['config']} -> {best['weighted_mean']}")
            except Exception as e:
                print(f"[DOCS][SKIP] {p.name} failed: {e}")

    if args.mode in ("hsi", "all"):
        hsi_files = list_mat_files(HSI_DIR)
        print(f"[HSI] found {len(hsi_files)} .mat files in {HSI_DIR}")
        for p in hsi_files:
            try:
                rows = ablation_one_dataset(
                    kind="hsi",
                    dataset_path=p,
                    seeds=args.seeds,
                    maxiter=args.maxiter,
                    max_features=None,
                    make_plots=args.plots,
                )
                all_rows.extend(rows)
                best = min(rows, key=lambda r: r["weighted_mean"] if r["weighted_mean"] is not None else np.inf)
                print(f"  - {p.name}: best(weighted_mean) = {best['config']} -> {best['weighted_mean']}")
            except Exception as e:
                print(f"[HSI][SKIP] {p.name} failed: {e}")

    out_csv = OUT_DIR / "summary_all.csv"
    write_csv(all_rows, out_csv)
    print(f"[OK] wrote summary CSV: {out_csv}")

    # petit tableau console: où γ+inertie aide le plus sur weighted
    # (delta_weighted_vs_pure < 0 = mieux)
    candidates = [r for r in all_rows if r["config"].startswith("γ + inertia")]
    candidates = [r for r in candidates if r.get("delta_weighted_vs_pure") is not None]
    candidates.sort(key=lambda r: r["delta_weighted_vs_pure"])
    print("\nTop-10 datasets where (γ+inertia) improves WEIGHTED objective the most (negative is good):")
    for r in candidates[:10]:
        print(f"  {r['kind']:4s} | {r['dataset']:<20s} | Δweighted={r['delta_weighted_vs_pure']:.6g} | Δacc={r['delta_acc_vs_pure']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
