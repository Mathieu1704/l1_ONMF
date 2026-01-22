import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd  # Pour un joli tableau
import seaborn as sns # Pour la heatmap (pip install seaborn si besoin, sinon matplotlib suffit)

# --- SETUP PATH ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from l1_ONMF.datasets import load_doc_mat
    from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
    from l1_ONMF.metrics import clustering_accuracy_hungarian
except ImportError:
    if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0, str(SCRIPT_DIR))
    from l1_ONMF.datasets import load_doc_mat
    from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
    from l1_ONMF.metrics import clustering_accuracy_hungarian

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

# ==========================================
# PARAMÈTRES DE LA GRID SEARCH
# ==========================================
# Tu peux ajuster ces listes selon le temps que tu as
GAMMAS = [1.0, 0.5, 0.1, 0.05, 0.01]  # De "Standard" à "Très relaxé"
ETAS   = [1.0, 0.7, 0.5, 0.3]        # De "Pas d'inertie" à "Forte inertie"

DATASET_NAME = "classic.mat"
MAX_ITER = 50 
SEED = 42

def run_grid_search():
    # 1. Chargement & Preprocess
    data_path = SCRIPT_DIR / "data" / "docs" / DATASET_NAME
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "docs" / DATASET_NAME
    
    if not data_path.exists():
        print(f"Erreur: {DATASET_NAME} introuvable.")
        return

    print(f"Chargement et préparation de {DATASET_NAME}...")
    X_raw, y_true, r = load_doc_mat(str(data_path))
    
    # Preprocess indispensable
    tfidf = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True)
    X = normalize(tfidf.fit_transform(X_raw.T).T, norm='l1', axis=0)

    results = []

    print(f"\nLancement de la Grid Search ({len(GAMMAS)} gammas x {len(ETAS)} etas)...")
    print("-" * 60)
    print(f"{'Gamma':<10} | {'Eta':<10} | {'ACCURACY':<10} | {'Pure Err':<12}")
    print("-" * 60)

    # 2. Boucle sur les paramètres
    for g in GAMMAS:
        for e in ETAS:
            opts = L1ONMFOptions(
                r=r,
                maxiter=MAX_ITER,
                verbose=False,
                seed=SEED,
                zero_weight=float(g),
                inertia_eta=float(e),
                init="kmeans",
                n_init=1 
            )
            
            # Run
            _, H, info = alternating_l1_onmf(X, opts)
            
            # Metric
            y_pred = np.argmax(H, axis=0) + 1
            acc = clustering_accuracy_hungarian(y_true, y_pred) * 100
            final_err = info['rel_l1_errors'][-1]
            
            print(f"{g:<10} | {e:<10} | {acc:.2f}%     | {final_err:.4f}")
            
            results.append({
                "gamma": g,
                "eta": e,
                "accuracy": acc,
                "final_error": final_err
            })

    # 3. Analyse des résultats
    df = pd.DataFrame(results)
    
    print("\n" + "="*30)
    print(" TOP 5 CONFIGURATIONS")
    print("="*30)
    print(df.sort_values("accuracy", ascending=False).head(5).to_string(index=False))

    # 4. Génération de la Heatmap (Indispensable pour le rapport)
    # Pivot pour avoir une matrice (Gammas en lignes, Etas en colonnes)
    pivot_table = df.pivot(index="gamma", columns="eta", values="accuracy")
    
    plt.figure(figsize=(8, 6))
    # Si tu n'as pas seaborn, commente les 2 lignes suivantes et utilise imshow
    try:
        import seaborn as sns
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': 'Accuracy (%)'})
        plt.title(f"Accuracy en fonction de Gamma et Eta\n({DATASET_NAME})")
    except ImportError:
        # Fallback matplotlib simple
        plt.imshow(pivot_table, cmap="viridis", aspect='auto')
        plt.colorbar(label='Accuracy (%)')
        plt.title("Installe seaborn pour une plus belle heatmap (`pip install seaborn`)")

    plt.ylabel("Gamma (Poids des zéros)")
    plt.xlabel("Eta (Inertie)")
    plt.tight_layout()
    
    out_img = SCRIPT_DIR / "grid_search_heatmap.png"
    plt.savefig(out_img, dpi=150)
    print(f"\nHeatmap sauvegardée : {out_img}")
    plt.show()

if __name__ == "__main__":
    run_grid_search()