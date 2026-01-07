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
N_INIT_BENCH = 1    

def preprocess_data_sparse(X_raw):
    """Reste en format sparse pour la cohérence de la régularisation."""
    tfidf = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True)
    X_t = tfidf.fit_transform(X_raw.T) # X_t est (docs x words) sparse
    
    if X_t.shape[1] > MAX_FEATURES:
        word_scores = np.asarray(X_t.sum(axis=0)).ravel()
        idx = np.argsort(word_scores)[-MAX_FEATURES:]
        X_t = X_t[:, idx]
    
    X = X_t.T # (words x docs)
    X = normalize(X, norm='l1', axis=0) # Reste sparse
    return X # Retourne une matrice scipy sparse

def run_dataset(filename):
    path = DATA_DIR / filename
    print(f"\n>>> Traitement de {filename}...")
    
    try:
        X_raw, y_true, r = load_doc_mat(str(path))
        X = preprocess_data_sparse(X_raw)
        m, n = X.shape
        print(f"    Data Sparse: {m} features, {n} docs, k={r}")
    except Exception as e:
        print(f"    [SKIP] Erreur: {e}")
        return None

    res = {"dataset": filename, "m": m, "n": n, "r": r}

    # --- ALGO 1: L1-ONMF (Nous) ---
    # On force n_init=1 pour être équitable avec les autres
    print("    Running L1-ONMF...", end="", flush=True)
    try:
        t0 = time.time()
        opts = L1ONMFOptions(r=r, maxiter=MAX_ITER, init="kmeans", n_init=N_INIT_BENCH, verbose=False, enforce_W_nonneg=True)
        _, H_l1, info = alternating_l1_onmf(X, opts)
        acc = clustering_accuracy_hungarian(y_true, np.argmax(H_l1, axis=0)+1)
        res["acc_L1"], res["time_L1"], res["it_L1"] = acc*100, time.time()-t0, info["num_iter"]
        print(f" Done ({acc*100:.1f}%)")
    except Exception as e: print(f" Failed ({e})")

    # --- ALGO 2: KL-ONMF (Prof) ---
    # Note: On doit souvent densifier pour le code du prof s'il n'est pas optimisé sparse
    print("    Running KL-ONMF...", end="", flush=True)
    try:
        t0 = time.time()
        # On passe X.toarray() ici seulement car le code KL du prof n'aime pas le sparse
        _, H_kl, _ = alternating_kl_onmf(X.toarray(), r=r, maxiter=MAX_ITER, init="kmeans", seed=42)
        acc = clustering_accuracy_hungarian(y_true, np.argmax(H_kl, axis=0)+1)
        res["acc_KL"], res["time_KL"], res["it_KL"] = acc*100, time.time()-t0, MAX_ITER
        print(f" Done ({acc*100:.1f}%)")
    except Exception as e: print(f" Failed ({e})")

    # --- ALGO 3: Fro-ONMF (KMeans) ---
    print("    Running Fro-ONMF...", end="", flush=True)
    try:
        t0 = time.time()
        # KMeans de sklearn gère très bien les matrices sparse en entrée
        kmeans = KMeans(n_clusters=r, n_init=N_INIT_BENCH, max_iter=MAX_ITER, random_state=42)
        y_pred = kmeans.fit_predict(X.T) + 1
        acc = clustering_accuracy_hungarian(y_true, y_pred)
        res["acc_Fro"], res["time_Fro"], res["it_Fro"] = acc*100, time.time()-t0, kmeans.n_iter_
        print(f" Done ({acc*100:.1f}%)")
    except Exception as e: print(f" Failed ({e})")

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