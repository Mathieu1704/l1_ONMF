# experiments/compare_real_docs.py
import time
import csv
import sys
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

# --- SETUP PATH ---
PKG_PARENT = Path(__file__).resolve().parents[2]
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from l1_ONMF.datasets import load_doc_mat
from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
from l1_ONMF.kl_onmf import alternating_kl_onmf
from l1_ONMF.metrics import clustering_accuracy_hungarian

# --- CONFIGURATION ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"
OUT_CSV = "table_comparative_real_docs.csv"

DATASETS = [
    "classic.mat", "sports.mat", "reviews.mat", "hitech.mat", "ohscal.mat",
    "la1.mat", "k1b.mat", "la12.mat", "la2.mat", "tr11.mat",
    "tr23.mat", "tr41.mat", "tr45.mat", "NG20.mat", "ng3sim.mat"
]

# --- FAIR / REPRO SETTINGS ---
MAX_FEATURES = 5000
MAX_ITER = 100          # IMPORTANT: same max iterations for all methods
N_INIT_BENCH = 1
SEED = 42               # same seed for all methods
ENFORCE_W_NONNEG_DOCS = True  # keep True for docs to match KL setting / interpretability

def preprocess_data_sparse(X_raw, max_features=MAX_FEATURES, seed=SEED):
    """
    Keep sparse throughout; TF-IDF then L1-normalize columns.
    IMPORTANT: we must use the same preprocessing for all methods.
    """
    tfidf = TfidfTransformer(norm="l1", use_idf=True, smooth_idf=True)
    X_t = tfidf.fit_transform(X_raw.T)  # (docs x words), sparse

    # Feature selection: top-k by total tf-idf mass (deterministic)
    if X_t.shape[1] > max_features:
        word_scores = np.asarray(X_t.sum(axis=0)).ravel()
        # argsort is deterministic; tie-breaking depends on numpy but is stable for same env
        idx = np.argsort(word_scores)[-max_features:]
        X_t = X_t[:, idx]

    X = X_t.T  # (words x docs)
    X = normalize(X, norm="l1", axis=0)  # sparse
    return X

def run_dataset(filename):
    path = DATA_DIR / filename
    print(f"\n=== Processing {filename} ===")

    try:
        X_raw, y_true, r = load_doc_mat(str(path))
        X = preprocess_data_sparse(X_raw)
        m, n = X.shape
        print(f"  Feature Selection: Top-{m} (was {X_raw.shape[0]})")
        print(f"  Data: m={m} n={n} r={r} (sparse)")
    except Exception as e:
        print(f"  [SKIP] Error loading/preprocessing: {e}")
        return None

    res = {"dataset": filename, "m": int(m), "n": int(n), "r": int(r)}

    # --- 1) L1-ONMF (ours) ---
    print("  Running L1-ONMF...", end="", flush=True)
    try:
        t0 = time.time()
        opts = L1ONMFOptions(
            r=r,
            maxiter=MAX_ITER,
            init="kmeans",
            n_init=N_INIT_BENCH,
            seed=SEED,                 # <-- IMPORTANT: same seed
            verbose=False,
            enforce_W_nonneg=ENFORCE_W_NONNEG_DOCS,  # docs: True recommended
        )
        _, H_l1, info = alternating_l1_onmf(X, opts)
        # H is (r x n) and hard; argmax gives 0..r-1
        y_pred = np.argmax(H_l1, axis=0) + 1
        acc = clustering_accuracy_hungarian(y_true, y_pred)

        it_l1 = info.get("num_iter", MAX_ITER)
        res["acc_L1"] = float(acc * 100)
        res["time_L1"] = float(time.time() - t0)
        res["it_L1"] = int(it_l1)
        print(f" Done (acc={acc*100:.2f}%, it={it_l1})")
    except Exception as e:
        print(f" Failed ({e})")
        res["acc_L1"], res["time_L1"], res["it_L1"] = None, None, None

    # --- 2) KL-ONMF (reference code) ---
    print("  Running KL-ONMF...", end="", flush=True)
    try:
        t0 = time.time()
        # KL code usually expects dense; keep preprocessing identical, only densify at the last moment.
        _, H_kl, info_kl = alternating_kl_onmf(
            X.toarray(),
            r=r,
            maxiter=MAX_ITER,
            init="kmeans",
            seed=SEED,   # <-- IMPORTANT: same seed
        )
        y_pred = np.argmax(H_kl, axis=0) + 1
        acc = clustering_accuracy_hungarian(y_true, y_pred)

        # If info_kl has an iteration count, prefer it; otherwise MAX_ITER.
        it_kl = info_kl.get("num_iter", MAX_ITER) if isinstance(info_kl, dict) else MAX_ITER
        res["acc_KL"] = float(acc * 100)
        res["time_KL"] = float(time.time() - t0)
        res["it_KL"] = int(it_kl)
        print(f" Done (acc={acc*100:.2f}%, it={it_kl})")
    except Exception as e:
        print(f" Failed ({e})")
        res["acc_KL"], res["time_KL"], res["it_KL"] = None, None, None

    # --- 3) Fro-ONMF baseline (KMeans) ---
    print("  Running Fro-ONMF (KMeans)...", end="", flush=True)
    try:
        t0 = time.time()
        kmeans = KMeans(
            n_clusters=r,
            n_init=N_INIT_BENCH,
            max_iter=MAX_ITER,
            random_state=SEED,  # <-- IMPORTANT: same seed
        )
        y_pred = kmeans.fit_predict(X.T) + 1
        acc = clustering_accuracy_hungarian(y_true, y_pred)

        res["acc_Fro"] = float(acc * 100)
        res["time_Fro"] = float(time.time() - t0)
        # KMeans stops early; that's fine—report the actual iterations.
        res["it_Fro"] = int(getattr(kmeans, "n_iter_", MAX_ITER))
        print(f" Done (acc={acc*100:.2f}%, it={res['it_Fro']})")
    except Exception as e:
        print(f" Failed ({e})")
        res["acc_Fro"], res["time_Fro"], res["it_Fro"] = None, None, None

    return res

def main():
    headers = [
        "dataset", "m", "n", "r",
        "acc_Fro", "acc_KL", "acc_L1",
        "time_Fro", "time_KL", "time_L1",
        "it_Fro", "it_KL", "it_L1",
    ]

    results = []
    for name in DATASETS:
        row = run_dataset(name)
        if row is not None:
            results.append(row)

    if results:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for row in results:
                w.writerow(row)

        print(f"\nSaved: {OUT_CSV}")
        # quick sanity summary
        l1_accs = [r["acc_L1"] for r in results if r.get("acc_L1") is not None]
        if l1_accs:
            print(f"Avg acc_L1 = {np.mean(l1_accs):.2f}% over {len(l1_accs)} datasets")

if __name__ == "__main__":
    main()
