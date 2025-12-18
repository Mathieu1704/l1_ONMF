import time
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------- PARAMÈTRES ----------
USE_TOY = False        # True pour tester la 10x10, False pour classic.mat
DATASET_NAME = "classic.mat"
MAXITER = 50
TOL = 1e-6
SEED = 0
DIM = 2000
# -------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l1_onmf import alternating_l1_onmf, L1ONMFOptions
from metrics import clustering_accuracy_hungarian, ari, nmi
from datasets import load_doc_mat  


def make_toy_matrix(m: int, n: int, r: int = 3, noise: float = 0.05, seed: int = 0):
    """
    Génère une petite matrice X = W_true @ H_true + bruit,
    avec des clusters "propres".
    """
    rng = np.random.default_rng(seed)

    # W_true : "centroïdes" positifs
    W_true = np.abs(rng.normal(size=(m, r)))

    # H_true : chaque colonne appartient à un seul cluster, avec une échelle positive
    H_true = np.zeros((r, n))
    y_true = np.zeros(n, dtype=int)
    for j in range(n):
        c = rng.integers(0, r)                      # cluster 0..r-1
        s = np.abs(rng.normal(loc=1.0, scale=0.2))  # échelle > 0
        H_true[c, j] = s
        y_true[j] = c + 1                           # labels 1..r pour compat avec métriques

    # Matrice observée X
    X = W_true @ H_true + noise * rng.normal(size=(m, n))

    return X, y_true, r, W_true, H_true


def run_on_matrix(X, y_true, r, label: str):
    """Facteur X, affiche les logs et trace la convergence."""
    print(f"X shape = {X.shape}, r = {r}  ({label})")

    # options de l'algo
    opts = L1ONMFOptions(
        r=r,
        maxiter=MAXITER,
        l1_tol=TOL,
        patience=3,
        seed=SEED,
        verbose=True,      
        log_errors=True,   # stocke les erreurs relatives L1
        init="warm_fro",
    )

    # Lancer l'algo
    t0 = time.perf_counter()
    W, H, info = alternating_l1_onmf(X, opts)
    t1 = time.perf_counter()
    print(f"\nTemps total = {t1 - t0:.4f} s, iters = {info.get('num_iter')}")

    # Clustering induit par H
    y_pred = np.asarray(H).argmax(axis=0) + 1
    acc = clustering_accuracy_hungarian(y_true, y_pred)
    print(f"Accuracy  = {acc*100:.2f}%")

    if not USE_TOY:
        print(f"ARI  = {ari(y_true, y_pred):.4f}")
        print(f"NMI  = {nmi(y_true, y_pred):.4f}")

    
    if USE_TOY:
        print("\nW (estimé) =")
        with np.printoptions(precision=3, suppress=True, linewidth=200):
            print(np.asarray(W))

        print("\nH (estimé) =")
        with np.printoptions(precision=3, suppress=True, linewidth=200):
            print(np.asarray(H))

    else:
        print(f"\nW shape = {W.shape}, H shape = {H.shape} (matrices non affichées car trop grandes)")

    # # Tracer la courbe d'erreur relative L1
    # errs = info.get("rel_l1_errors", None)
    # if errs is not None and len(errs) > 0:
    #     iters = np.arange(1, len(errs) + 1)
    #     plt.figure()
    #     plt.plot(iters, errs, marker="o")
    #     plt.xlabel("Itération")
    #     plt.ylabel("Erreur relative L1")
    #     plt.title(f"Convergence L1-ONMF ({label})")
    #     plt.grid(True)
    #     plt.show()
    # else:
    #     print("Pas d'erreurs enregistrées dans info['rel_l1_errors'].")

    # Tracer la courbe d'erreur relative L1
    errs = info.get("rel_l1_errors", None)
    if errs is not None and len(errs) > 0:
        iters = np.arange(1, len(errs) + 1)
        plt.figure()
        plt.plot(iters, errs, marker="o")
        plt.xlabel("Itération")
        plt.ylabel("Erreur relative L1")
        plt.title(f"Convergence L1-ONMF ({label})")
        plt.grid(True)

        out = ROOT / "experiments" / "convergence.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Plot sauvegardé : {out}")
    else:
        print("Pas d'erreurs enregistrées dans info['rel_l1_errors'].")



def main():
    if USE_TOY:
        # ====== CAS MATRICE  ======
        X, y_true, r, W_true, H_true = make_toy_matrix(m=DIM, n=DIM, r=3, noise=0.05, seed=0)
        print(f">>> Test sur matrice synthétique {DIM}x{DIM}")
        print("W_true et H_true connus (pas nécessaire pour l'algo, mais pour le debug).")
        run_on_matrix(X, y_true, r, label=f"{DIM}x{DIM}")
        


    else:
        # ====== CAS classic.mat ======
        data_dir = ROOT / "data" / "docs"
        mat_path = data_dir / DATASET_NAME
        print(f">>> Test sur dataset réel : {mat_path}")

        X, y_true, r = load_doc_mat(str(mat_path))  # X peut être sparse, l1_onmf le gère
        run_on_matrix(X, y_true, r, label=DATASET_NAME)


if __name__ == "__main__":
    main()