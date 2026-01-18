# experiments/doc_experiments.py
import time, csv
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

# --- SETUP PATH ---
PKG_PARENT = Path(__file__).resolve().parents[2]
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "docs"

from l1_ONMF.datasets import load_doc_mat
from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
from l1_ONMF.metrics import clustering_accuracy_hungarian, ari, nmi

# ===== PARAMETRES =====
OUT_CSV = "results_optimized.csv"
MAXITER = 100 # On laisse le temps de converger avec l'inertie
TOL = 1e-6
SEED = 42
N_INIT = 5    # On assure le coup avec 5 inits

DATASETS = [
    "classic.mat", "sports.mat", "reviews.mat", "hitech.mat", "ohscal.mat", 
    "la1.mat", "k1b.mat", "la12.mat", "la2.mat", "tr11.mat", 
    "tr23.mat", "tr41.mat", "tr45.mat", "NG20.mat", "ng3sim.mat"
]

def save_convergence_plot(dataset_name, info, seed):
    """Affiche les deux courbes : Pure (réelle) et Weighted (optimisée)."""
    errors_pure = info.get("rel_l1_errors")
    errors_weighted = info.get("rel_l1_errors_weighted")
    
    if errors_pure is None or len(errors_pure) == 0: return

    plt.figure(figsize=(10, 6))
    
    # 1. Courbe Pure (L'erreur mathématique stricte)
    plt.plot(errors_pure, marker='o', markersize=3, label='L1 Pure (Real)', alpha=0.6)
    
    # 2. Courbe Weighted (Ce que l'algo optimise vraiment)
    if errors_weighted is not None and len(errors_weighted) > 0:
        plt.plot(errors_weighted, marker='x', markersize=3, label='L1 Weighted (Optimized)', linestyle='--')

    plt.title(f"Convergence: {dataset_name} (seed={seed})")
    plt.xlabel("Iterations")
    plt.ylabel("Relative Error")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Sauvegarde
    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{dataset_name}_conv.png", dpi=150)
    plt.close()

def run_one(path):
    dataset_name = Path(path).name
    print(f"\n=== Processing {dataset_name} ===")
    
    try:
        X_raw, y, r = load_doc_mat(path)
    except Exception as e:
        print(f"[ERROR] Loading failed (corrupt file?): {e}")
        return None

    # 1. TF-IDF
    X_t = X_raw.T
    tfidf = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True)
    X_t = tfidf.fit_transform(X_t)
    
    # 2. SELECTION PLUS LARGE (Répond à ta question sur m=1000)
    # On monte à 5000 pour avoir plus de "corps", tout en éliminant le bruit extrême
    K_FEATURES = 5000
    if X_t.shape[1] > K_FEATURES:
        word_scores = np.asarray(X_t.sum(axis=0)).ravel()
        top_indices = np.argsort(word_scores)[-K_FEATURES:]
        X_t = X_t[:, top_indices]
        print(f"  Feature Selection: Top-{K_FEATURES} (was {X_raw.shape[0]})")
    else:
        print(f"  Kept all {X_t.shape[1]} features.")
    
    X = X_t.T # (m, n)

    # 3. NORMALISATION CRITIQUE$$
    # Indispensable pour que K-Means (l'init) et L1 (l'algo) voient la même géométrie
    X = normalize(X, norm='l1', axis=0)

    opts = L1ONMFOptions(
        r=r,
        maxiter=MAXITER,
        l1_tol=TOL, 
        patience=20, # Patience élevée car l'inertie ralentit la convergence mais la rend stable
        seed=SEED,
        verbose=True,
        log_errors=True,
        enforce_W_nonneg=False,
        init="kmeans",  
        n_init=N_INIT,
    )

    t0 = time.perf_counter()
    W, H, info = alternating_l1_onmf(X, opts)   
    t1 = time.perf_counter()

    c_pred = np.asarray(H).argmax(axis=0) + 1
    acc = clustering_accuracy_hungarian(y, c_pred)
    
    print(f"  => ACC: {acc*100:.2f}% | Time: {t1-t0:.2f}s")
    save_convergence_plot(dataset_name, info, info.get("seed"))

    return {
        "dataset": dataset_name,
        "m": X.shape[0],
        "n": X.shape[1],
        "r": r,
        "acc": acc,
        "ari": ari(y, c_pred),
        "nmi": nmi(y, c_pred),
        "time_s": t1 - t0,
        "iters": info.get("num_iter", None),    
    }

def main():
    rows = []
    for name in DATASETS:
        metrics = run_one(str(DATA_DIR / name))
        if metrics: rows.append(metrics)

    if rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["dataset","m","n","r","acc","ari","nmi","time_s","iters"])
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nResults written to {OUT_CSV}")

if __name__ == "__main__":
    main()