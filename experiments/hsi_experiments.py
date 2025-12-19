# experiments/hsi_experiments.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import csv
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# --- SETUP PATH ---
PKG_PARENT = Path(__file__).resolve().parents[2]
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
from l1_ONMF.kl_onmf import alternating_kl_onmf
from l1_ONMF.datasets import load_hsi_mat

# --- CONFIGURATION ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hsi"
OUT_DIR = Path(__file__).resolve().parent / "plots" / "hsi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = "table_hsi_results.csv"

DATASETS = ["Moffet", "Samson", "Jasper"]

# ==========================================
# 1. OUTILS MATHÉMATIQUES (MRSA & SCALING)
# ==========================================

def compute_mrsa(x, y):
    """Calcule le Mean Removed Spectral Angle entre deux vecteurs."""
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    nom = np.dot(x_c, y_c)
    denom = np.linalg.norm(x_c) * np.linalg.norm(y_c) + 1e-16
    val = np.clip(nom / denom, -1.0, 1.0)
    return (100 / np.pi) * np.arccos(val)

def match_and_score_mrsa(W_true, W_pred):
    """
    Trouve la meilleure permutation des colonnes de W_pred pour coller à W_true.
    Retourne: score MRSA moyen, W_pred permuté, indices de permutation
    """
    if W_true is None: return 0.0, W_pred, np.arange(W_pred.shape[1])
    
    r = W_true.shape[1]
    if W_pred.shape[1] != r: return 99.9, W_pred, np.arange(W_pred.shape[1])

    cost_mat = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            cost_mat[i, j] = compute_mrsa(W_true[:, i], W_pred[:, j])
            
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    W_aligned = W_pred[:, col_ind]
    avg_mrsa = cost_mat[row_ind, col_ind].mean()
    
    return avg_mrsa, W_aligned, col_ind

def scale_W_to_GroundTruth(W_pred, W_true):
    """
    Ajuste l'amplitude de W_pred sur celle de W_true pour que les plots soient superposables.
    """
    if W_true is None: return W_pred
    W_scaled = W_pred.copy()
    r = W_pred.shape[1]
    for k in range(r):
        max_pred = np.max(W_pred[:, k]) + 1e-16
        max_true = np.max(W_true[:, k])
        W_scaled[:, k] = (W_pred[:, k] / max_pred) * max_true
    return W_scaled

# ==========================================
# 2. FONCTIONS DE PLOT (Signatures, Color, N&B)
# ==========================================

def get_robust_image_dims(n_pixels, dimx, dimy):
    """Calcul sécurisé des dimensions de l'image (Safety Net)."""
    if dimx is None or dimy is None or dimx * dimy != n_pixels:
        side = int(np.sqrt(n_pixels))
        return side, side, True # True = a été forcé/tronqué
    return dimx, dimy, False

def plot_all_signatures(W_dict, dataset_name):
    """Génère une figure avec les signatures spectrales alignées."""
    n_methods = len(W_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 3.5), sharey=True)
    if n_methods == 1: axes = [axes]
    
    for ax, (method_name, W) in zip(axes, W_dict.items()):
        ax.plot(W)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if method_name == "Ground Truth":
            ax.set_ylabel("Reflectance")
            
    plt.suptitle(f"{dataset_name} - Spectral Signatures", y=1.05)
    plt.tight_layout()
    save_path = OUT_DIR / f"{dataset_name}_ALL_Signatures.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_colored_comparison(H_dict, dimx, dimy, dataset_name):
    """Génère une ligne d'images colorées (Segmentation Map)."""
    methods = list(H_dict.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(3 * n_methods, 3.5), constrained_layout=True)
    if n_methods == 1: axes = [axes]
    
    # Palette fixe pour cohérence entre méthodes
    r = H_dict[methods[0]].shape[0]
    colors = ['#1f77b4', '#2ca02c', '#8c564b', '#7f7f7f', '#d62728', '#17becf']
    cmap = ListedColormap(colors[:r]) if r <= len(colors) else 'tab10'

    for ax, (method_name, H) in zip(axes, H_dict.items()):
        labels = np.argmax(H, axis=0)
        final_dimx, final_dimy, truncated = get_robust_image_dims(labels.size, dimx, dimy)
        if truncated: labels = labels[:final_dimx*final_dimy]

        try:
            img = labels.reshape((final_dimy, final_dimx), order='F')
        except:
            img = labels.reshape((final_dimy, final_dimx), order='C')
            
        ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Cadre noir
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor('black'); spine.set_linewidth(1.5)

    plt.suptitle(f"{dataset_name} - Segmentation", y=1.05)
    plt.savefig(OUT_DIR / f"{dataset_name}_Colored_Maps.png", bbox_inches='tight', dpi=150)
    plt.close()

def plot_decomposition_maps(H_dict, dimx, dimy, dataset_name):
    """Génère la grille N&B décomposée par matériau (Comme Fig 2 du papier)."""
    methods = list(H_dict.keys())
    n_methods = len(methods)
    r = H_dict[methods[0]].shape[0]
    
    fig, axes = plt.subplots(n_methods, r, figsize=(2.5 * r, 2.5 * n_methods), constrained_layout=True)
    if n_methods == 1: axes = axes[None, :]
    if r == 1: axes = axes[:, None]

    for row_idx, (method_name, H) in enumerate(H_dict.items()):
        labels = np.argmax(H, axis=0)
        final_dimx, final_dimy, truncated = get_robust_image_dims(labels.size, dimx, dimy)
        if truncated: labels = labels[:final_dimx*final_dimy]

        for k in range(r):
            ax = axes[row_idx, k]
            binary_map = (labels == k).astype(int)
            
            try:
                img = binary_map.reshape((final_dimy, final_dimx), order='F')
            except:
                img = binary_map.reshape((final_dimy, final_dimx), order='C')
            
            # 0=Blanc, 1=Noir
            ax.imshow(img, cmap='Greys', interpolation='nearest', vmin=0, vmax=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            
            # Cadre fin
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_edgecolor('black'); spine.set_linewidth(0.8)

            if row_idx == 0: ax.set_title(f"Material {k+1}", fontsize=10)
            if k == 0: ax.set_ylabel(method_name, fontsize=11, fontweight='bold')

    plt.savefig(OUT_DIR / f"{dataset_name}_Decomposition.png", bbox_inches='tight', dpi=150)
    plt.close()

# ==========================================
# 3. MOTEUR PRINCIPAL (BENCHMARK)
# ==========================================

def run_hsi_bench():
    results = []
    print("=== LANCEMENT BENCHMARK HSI (L1 vs KL vs Fro) ===\n")
    
    for name in DATASETS:
        mat_path = DATA_DIR / f"{name}.mat"
        if not mat_path.exists():
            print(f"[SKIP] {name}.mat introuvable")
            continue
            
        print(f">>> Traitement de {name}...")
        
        # 1. Chargement & Robustesse
        try:
            X, r, W_true, dimx, dimy = load_hsi_mat(str(mat_path))
            
            # Force r si manquant
            if r is None:
                r = 3 if name in ["Moffet", "Samson"] else 4
                print(f"    (Warning: r forcé à {r})")
            
            # --- CORRECTION FINALE : PAS DE NORMALISATION DU GROUND TRUTH ---
            # On laisse le Ground Truth "brut". 
            # Comme ça, l'eau restera sombre (valeurs faibles) et ne sera pas étirée à 1.
            # Seuls les W négatifs sont clippés à 0 par sécurité.
            if W_true is not None:
                W_true = np.maximum(0, W_true)
                # SUPPRIMÉ : W_true = W_true / norms (C'était la cause du bug)
                
        except Exception as e:
            print(f"    Erreur chargement: {e}")
            continue
            
        res = {"dataset": name, "m": X.shape[0], "n": X.shape[1], "r": r}
        
        W_store = {}
        H_store = {}
        
        # 2. Construction Ground Truth H (via Cosine Similarity)
        if W_true is not None:
            W_store["Ground Truth"] = W_true
            
            W_n = W_true / (np.linalg.norm(W_true, axis=0, keepdims=True) + 1e-16)
            X_n = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-16)
            labels_gt = np.argmax(W_n.T @ X_n, axis=0)
            
            H_true = np.zeros((r, X.shape[1]))
            for i in range(X.shape[1]): H_true[labels_gt[i], i] = 1.0
            H_store["Ground Truth"] = H_true

        # 3. Fro-ONMF (K-Means)
        t0 = time.time()
        kmeans = KMeans(n_clusters=r, n_init=5, max_iter=100, random_state=42)
        labels_fro = kmeans.fit_predict(X.T)
        t_fro = time.time() - t0
        
        H_fro = np.zeros((r, X.shape[1]))
        for i in range(X.shape[1]): H_fro[labels_fro[i], i] = 1
        W_fro = kmeans.cluster_centers_.T
        
        score_fro, W_fro_aligned, col_ind_fro = match_and_score_mrsa(W_true, W_fro)
        W_fro_scaled = scale_W_to_GroundTruth(W_fro_aligned, W_true)
        
        W_store["Fro-ONMF"] = W_fro_scaled
        H_store["Fro-ONMF"] = H_fro[col_ind_fro, :] # Alignement H
        res["MRSA_Fro"] = score_fro
        print(f"    [Fro] MRSA: {score_fro:.2f} | Time: {t_fro:.2f}s")

        # 4. KL-ONMF
        t0 = time.time()
        W_kl_raw, H_kl_raw, _ = alternating_kl_onmf(X, r, maxiter=50, init="kmeans", seed=42)
        t_kl = time.time() - t0
        
        score_kl, W_kl_aligned, col_ind_kl = match_and_score_mrsa(W_true, W_kl_raw)
        W_kl_scaled = scale_W_to_GroundTruth(W_kl_aligned, W_true)
        
        W_store["KL-ONMF"] = W_kl_scaled
        H_store["KL-ONMF"] = H_kl_raw[col_ind_kl, :]
        res["MRSA_KL"] = score_kl
        print(f"    [KL ] MRSA: {score_kl:.2f} | Time: {t_kl:.2f}s")

        # 5. L1-ONMF (Nous)
        t0 = time.time()
        opts = L1ONMFOptions(r=r, maxiter=50, init="kmeans", verbose=False, enforce_W_nonneg=True)
        W_l1_raw, H_l1_raw, info_l1 = alternating_l1_onmf(X, opts)
        t_l1 = time.time() - t0
        
        score_l1, W_l1_aligned, col_ind_l1 = match_and_score_mrsa(W_true, W_l1_raw)
        W_l1_scaled = scale_W_to_GroundTruth(W_l1_aligned, W_true)
        
        W_store["L1-ONMF"] = W_l1_scaled
        H_store["L1-ONMF"] = H_l1_raw[col_ind_l1, :]
        res["MRSA_L1"] = score_l1
        print(f"    [L1 ] MRSA: {score_l1:.2f} | Time: {t_l1:.2f}s")
        
        results.append(res)
        
        # 6. Plots
        plot_all_signatures(W_store, name)
        plot_colored_comparison(H_store, dimx, dimy, name)
        plot_decomposition_maps(H_store, dimx, dimy, name)
        print(f"    -> Images sauvegardées dans {OUT_DIR}")

    # 7. Sauvegarde CSV
    if results:
        headers = ["dataset", "m", "n", "r", "MRSA_Fro", "MRSA_KL", "MRSA_L1"]
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for row in results:
                w.writerow({k: row.get(k, "") for k in headers})
        print(f"\nTableau complet sauvegardé : {OUT_CSV}")

if __name__ == "__main__":
    run_hsi_bench()