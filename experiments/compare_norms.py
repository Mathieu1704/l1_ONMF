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

# Liste des datasets du papier
DATASETS = [
    "classic.mat", "sports.mat", "reviews.mat", "hitech.mat", "ohscal.mat", 
    "la1.mat", "k1b.mat", "la12.mat", "la2.mat", "tr11.mat", 
    "tr23.mat", "tr41.mat", "tr45.mat", "NG20.mat", "ng3sim.mat"
]

# Paramètres globaux pour que ce soit rapide mais significatif
MAX_FEATURES = 5000  # On ne garde que les 5000 mots les plus fréquents (Vital pour la vitesse)
MAX_ITER = 50        # 50 itérations suffisent pour voir la tendance
N_INIT = 1           # 1 seul essai pour le tableau (sinon c'est trop long)

def preprocess_data(X_raw):
    """Pipeline standardisé pour être juste avec tout le monde."""
    # 1. TF-IDF
    tfidf = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True)
    X_t = tfidf.fit_transform(X_raw.T)
    
    # 2. Feature Selection (Top-K)
    if X_t.shape[1] > MAX_FEATURES:
        word_scores = np.asarray(X_t.sum(axis=0)).ravel()
        idx = np.argsort(word_scores)[-MAX_FEATURES:]
        X_t = X_t[:, idx]
    
    # 3. Transpose (m x n) et Normalisation L1
    X = X_t.T
    X = normalize(X, norm='l1', axis=0)
    
    # Conversion Dense pour simplifier les algos (avec 5000 feats ça passe en RAM)
    return X.toarray()

def run_dataset(filename):
    path = DATA_DIR / filename
    print(f"\n>>> Traitement de {filename}...")
    
    # 1. Chargement
    try:
        X_raw, y_true, r = load_doc_mat(str(path))
    except Exception as e:
        print(f"    [SKIP] Erreur chargement: {e}")
        return None

    m_orig, n = X_raw.shape
    
    # 2. Preprocessing
    try:
        X = preprocess_data(X_raw)
        m = X.shape[0]
        print(f"    Data: m={m} (was {m_orig}), n={n}, k={r}")
    except Exception as e:
        print(f"    [SKIP] Erreur preprocessing: {e}")
        return None

    res = {
        "dataset": filename, "m": m, "n": n, "r": r,
        "acc_L1": 0, "time_L1": 0, "it_L1": 0,
        "acc_KL": 0, "time_KL": 0, "it_KL": 0,
        "acc_Fro": 0, "time_Fro": 0, "it_Fro": 0
    }

    # --- ALGO 1: L1-ONMF (Nous) ---
    print("    Running L1-ONMF...", end="", flush=True)
    try:
        t0 = time.time()
        opts = L1ONMFOptions(r=r, maxiter=MAX_ITER, init="kmeans", verbose=False, enforce_W_nonneg=True)
        _, H_l1, info = alternating_l1_onmf(X, opts)
        acc = clustering_accuracy_hungarian(y_true, np.argmax(H_l1, axis=0)+1)
        res["acc_L1"] = acc * 100
        res["time_L1"] = time.time() - t0
        res["it_L1"] = info["num_iter"]
        print(f" Done ({acc*100:.1f}%)")
    except Exception as e:
        print(f" Failed ({e})")

    # --- ALGO 2: KL-ONMF (Prof) ---
    print("    Running KL-ONMF...", end="", flush=True)
    try:
        t0 = time.time()
        # On utilise notre portage Python exact
        _, H_kl, _ = alternating_kl_onmf(X, r=r, maxiter=MAX_ITER, init="kmeans", seed=42)
        acc = clustering_accuracy_hungarian(y_true, np.argmax(H_kl, axis=0)+1)
        res["acc_KL"] = acc * 100
        res["time_KL"] = time.time() - t0
        res["it_KL"] = MAX_ITER # Le portage simple fait maxiter fixe
        print(f" Done ({acc*100:.1f}%)")
    except Exception as e:
        print(f" Failed ({e})")

    # --- ALGO 3: Fro-ONMF (Proxy K-Means) ---
    print("    Running Fro-ONMF (KMeans)...", end="", flush=True)
    try:
        t0 = time.time()
        kmeans = KMeans(n_clusters=r, n_init=5, max_iter=MAX_ITER, random_state=42)
        y_pred = kmeans.fit_predict(X.T) + 1
        acc = clustering_accuracy_hungarian(y_true, y_pred)
        res["acc_Fro"] = acc * 100
        res["time_Fro"] = time.time() - t0
        res["it_Fro"] = kmeans.n_iter_
        print(f" Done ({acc*100:.1f}%)")
    except Exception as e:
        print(f" Failed ({e})")

    return res

def main():
    results = []
    
    # Headers du CSV
    headers = [
        "dataset", "m", "n", "r",
        "acc_Fro", "acc_KL", "acc_L1",
        "time_Fro", "time_KL", "time_L1",
        "it_Fro", "it_KL", "it_L1"
    ]
    
    # Exécution
    for name in DATASETS:
        r = run_dataset(name)
        if r: results.append(r)
        
    # Sauvegarde CSV
    if results:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for row in results: w.writerow(row)
        print(f"\nTableau sauvegardé dans: {OUT_CSV}")

if __name__ == "__main__":
    main()