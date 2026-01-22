import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- SETUP PATH ---
# Le script est dans .../Research Project/l1_ONMF/
# On doit ajouter .../Research Project/ au sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports du projet
try:
    from l1_ONMF.datasets import load_doc_mat
    from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
    from l1_ONMF.metrics import clustering_accuracy_hungarian
except ImportError:
    # Fallback au cas où le script serait déplacé à la racine
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from l1_ONMF.datasets import load_doc_mat
    from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
    from l1_ONMF.metrics import clustering_accuracy_hungarian

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

def run_comparison():
    # 1. Chargement (Classic.mat)
    # On cherche d'abord dans le dossier data relatif au script
    # Structure supposée: .../l1_ONMF/data/docs/classic.mat
    data_path = SCRIPT_DIR / "data" / "docs" / "classic.mat"
    
    # Si pas trouvé, on cherche un étage plus haut (si data est à côté de l1_ONMF)
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "docs" / "classic.mat"

    if not data_path.exists():
        print(f"[ERREUR] Impossible de trouver le fichier : classic.mat")
        print(f"Cherché ici : {data_path}")
        return

    print(f"Chargement de {data_path.name}...")
    try:
        X_raw, y_true, r = load_doc_mat(str(data_path))
    except Exception as e:
        print(f"Erreur chargement : {e}")
        return
    
    # --- PRÉTRAITEMENT ---
    print("Prétraitement (TF-IDF + Normalize)...")
    tfidf = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True)
    X_t = X_raw.T 
    X_tfidf = tfidf.fit_transform(X_t)
    X = X_tfidf.T 
    X = normalize(X, norm='l1', axis=0) # (m x n)

    m, n = X.shape
    print(f"Données: m={m}, n={n}, r={r}")

    # --- CONFIG 1: STANDARD ---
    print("\n--- 1. Standard (gamma=1.0) ---")
    opts_std = L1ONMFOptions(
        r=r, maxiter=50, verbose=False, seed=42,
        zero_weight=1.0, inertia_eta=1.0, 
        init="kmeans", n_init=1
    )
    _, H_std, info_std = alternating_l1_onmf(X, opts_std)
    y_pred_std = np.argmax(H_std, axis=0) + 1
    acc_std = clustering_accuracy_hungarian(y_true, y_pred_std)
    print(f"Standard ACC : {acc_std*100:.2f}%")

    # --- CONFIG 2: OPTIMISÉE ---
    print("\n--- 2. Optimisé (gamma=0.05) ---")
    opts_opt = L1ONMFOptions(
        r=r, maxiter=50, verbose=False, seed=42,
        zero_weight=0.05, inertia_eta=0.3,
        init="kmeans", n_init=1
    )
    _, H_opt, info_opt = alternating_l1_onmf(X, opts_opt)
    y_pred_opt = np.argmax(H_opt, axis=0) + 1
    acc_opt = clustering_accuracy_hungarian(y_true, y_pred_opt)
    
    print(f"Optimized ACC : {acc_opt*100:.2f}%")
    print(f"(Note: L'erreur L1 pure est plus élevée ici, c'est normal)")

    # --- PLOT ---
    print("\nGénération du plot...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(info_std['rel_l1_errors'], 'o-', label='Erreur L1 Pure')
    plt.title(f"Standard (gamma=1)\nACC={acc_std*100:.1f}%")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(info_opt['rel_l1_errors_weighted'], 'x--', label='Objectif Optimisé')
    plt.plot(info_opt['rel_l1_errors'], 'o-', alpha=0.3, label='Erreur L1 Pure')
    plt.title(f"Optimisé (gamma=0.05)\nACC={acc_opt*100:.1f}%")
    plt.grid(True); plt.legend()
    
    plt.tight_layout()
    out_file = SCRIPT_DIR / "comparison_plot.png"
    plt.savefig(out_file, dpi=150)
    print(f"Sauvegardé : {out_file}")
    plt.show()

if __name__ == "__main__":
    run_comparison()