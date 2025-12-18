# experiments/synthetic_benchmark.py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

# --- SETUP PATH ---
PKG_PARENT = Path(__file__).resolve().parents[2]
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
from l1_ONMF.metrics import clustering_accuracy_hungarian

def generate_data(type="dense_outliers", m=100, n=500, r=4, noise_level=0.1):
    """
    Génère des données synthétiques contrôlées.
    """
    rng = np.random.default_rng(42)
    
    # Vrais centroids W (m x r)
    W_true = np.abs(rng.standard_normal((m, r)))
    # Normalisation
    W_true = W_true / np.sum(W_true, axis=0)[None, :]
    
    # Vrais clusters H (r x n) - Hard Clustering
    H_true = np.zeros((r, n))
    labels_true = rng.integers(0, r, size=n)
    for j in range(n):
        # Scale variable
        H_true[labels_true[j], j] = 1.0 + 0.1 * np.abs(rng.standard_normal()) 
    
    # Matrice idéale
    X_clean = W_true @ H_true
    
    if type == "dense_gaussian":
        # Bruit normal (L2 est optimal ici, L1 devrait être OK)
        noise = rng.standard_normal((m, n))
        X = X_clean + noise_level * noise
        
    elif type == "dense_outliers":
        # Bruit impulsionnel (L1 DOIT GAGNER ICI)
        # On corrompt 10% des entrées avec de très grandes valeurs
        X = X_clean.copy()
        mask = rng.random((m, n)) < 0.1
        # Gros outliers (x 20)
        X[mask] += 20.0 * rng.random(np.sum(mask)) 
        
    elif type == "sparse_docs":
        # Simulation de documents (très creux)
        X = np.zeros((m, n))
        for j in range(n):
            k = labels_true[j]
            # Le doc j hérite des mots du topic k
            prob = W_true[:, k]
            prob = prob / prob.sum()
            # On tire 20 mots par document
            words = rng.choice(m, size=20, p=prob)
            for w in words:
                X[w, j] += 1.0
        
    else:
        raise ValueError("Unknown type")
        
    return np.maximum(0, X), labels_true + 1

def run_experiment(data_type, m, n, r):
    print(f"\n--- Benchmarking: {data_type} (m={m}, n={n}, r={r}) ---")
    X, y_true = generate_data(type=data_type, m=m, n=n, r=r)
    
    # Options L1-ONMF
    opts = L1ONMFOptions(
        r=r,
        maxiter=100,
        init="kmeans",
        verbose=False,
        enforce_W_nonneg=True,
        n_init=1 # Un seul essai pour aller vite sur le benchmark
    )
    
    # Run
    t0 = time.time()
    W, H, info = alternating_l1_onmf(X, opts)
    t1 = time.time()
    
    # Eval
    y_pred = np.argmax(H, axis=0) + 1
    acc = clustering_accuracy_hungarian(y_true, y_pred)
    
    print(f"Result: ACC = {acc*100:.2f}% | Time = {t1-t0:.2f}s")
    
    # Plot Convergence (Correction du chemin ici)
    if len(info["rel_l1_errors"]) > 0:
        plt.figure()
        plt.plot(info["rel_l1_errors"], '.-')
        plt.title(f"Convergence {data_type}\nACC={acc*100:.1f}%")
        plt.xlabel("Iter")
        plt.ylabel("Rel L1 Error")
        plt.grid(True)
        
        # Création dossier sécurisée
        out_dir = Path(__file__).parent / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = out_dir / f"bench_{data_type}.png"
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    # 1. Cas facile (Bruit Gaussien)
    run_experiment("dense_gaussian", m=50, n=500, r=4)
    
    # 2. Cas de force majeure (Outliers) -> C'est là que L1 doit briller
    run_experiment("dense_outliers", m=50, n=500, r=4)
    
    # 3. Cas difficile (Sparse) 
    run_experiment("sparse_docs", m=1000, n=1000, r=4)