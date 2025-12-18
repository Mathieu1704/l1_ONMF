# experiments/toy_experiments.py
# ==============================
# L1-ONMF sur une matrice synthétique (toy) SANS argparse.
# Lance simplement:
#     python experiments/toy_experiments.py

# ===== PARAMÈTRES =====
OUT_CSV  = "results_toy_snpa.csv"
MAXITER  = 50
TOL      = 1e-6
SEED     = 0

DIM      = 2000     # matrice DIM x DIM
R        = 3        # nb de clusters
NOISE    = 0.05
PATIENCE = 3

PLOT_CONV = True
# ======================

import time, csv
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- IMPORTANT: mettre le parent de l1_ONMF dans sys.path ---
PKG_PARENT = Path(__file__).resolve().parents[2]  # ...\Research Project
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

# Imports via le package (PAS en relatif local)
from l1_ONMF import alternating_l1_onmf, L1ONMFOptions
from l1_ONMF.metrics import clustering_accuracy_hungarian, ari, nmi


def make_toy_matrix(m: int, n: int, r: int, noise: float, seed: int):
    """
    Génère X = W_true @ H_true + bruit, avec labels y_true (1..r).
    H_true : hard clustering (1 seule valeur non nulle par colonne).
    """
    rng = np.random.default_rng(seed)

    W_true = np.abs(rng.normal(size=(m, r)))

    H_true = np.zeros((r, n))
    y_true = np.zeros(n, dtype=int)

    for j in range(n):
        c = rng.integers(0, r)                      # 0..r-1
        s = np.abs(rng.normal(loc=1.0, scale=0.2))  # échelle > 0
        H_true[c, j] = s
        y_true[j] = c + 1                           # labels 1..r

    # Matrice observée X (bruit non négatif pour permettre SNPA)
    X = W_true @ H_true + noise * np.abs(rng.normal(size=(m, n)))

    return X, y_true, W_true, H_true


def run_one():
    X, y_true, W_true, H_true = make_toy_matrix(
        m=DIM, n=DIM, r=R, noise=NOISE, seed=SEED
    )

    # Options "à la doc_experiments": tout centralisé ici
    opts = L1ONMFOptions(
        r=R,
        maxiter=MAXITER,
        l1_tol=TOL,
        patience=PATIENCE,
        seed=SEED,
        verbose=True,
        log_errors=True,
        enforce_W_nonneg=True,   # cohérent avec ta génération (W_true >= 0)
        init="snpa",
        n_init=3,
        init_prune_top=500,
    )

    print(f">>> Toy matrix {DIM}x{DIM}, r={R}, noise={NOISE}")
    print(f"X shape = {X.shape}")

    t0 = time.perf_counter()
    W, H, info = alternating_l1_onmf(X, opts)
    t1 = time.perf_counter()

    y_pred = np.asarray(H).argmax(axis=0) + 1

    # Sur toy : acc + (optionnellement) ARI/NMI, ça marche aussi
    metrics = {
        "dataset": f"toy_{DIM}x{DIM}",
        "m": X.shape[0],
        "n": X.shape[1],
        "r": R,
        "acc": clustering_accuracy_hungarian(y_true, y_pred),
        "ari": ari(y_true, y_pred),
        "nmi": nmi(y_true, y_pred),
        "time_s": t1 - t0,
        "iters": info.get("num_iter", None),
    }

    # Plot convergence (même logique que toi)
    if PLOT_CONV:
        errs = info.get("rel_l1_errors", None)
        if errs is not None and len(errs) > 0:
            iters = np.arange(1, len(errs) + 1)
            plt.figure()
            plt.plot(iters, errs, marker="o")
            plt.xlabel("Itération")
            plt.ylabel("Erreur relative L1")
            plt.title(f"Convergence L1-ONMF (toy {DIM}x{DIM})")
            plt.grid(True)

            out = Path(__file__).resolve().parents[1] / "experiments" / "convergence_toy.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Plot sauvegardé : {out}")
        else:
            print("Pas d'erreurs enregistrées dans info['rel_l1_errors'].")

    return metrics


def main():
    row = run_one()

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset","m","n","r","acc","ari","nmi","time_s","iters"])
        w.writeheader()
        w.writerow(row)

    print(f"\nRésultats écrits dans {OUT_CSV}")
    print(
        f"ACC={row['acc']*100:.2f}% | ARI={row['ari']:.4f} | NMI={row['nmi']:.4f} | "
        f"time={row['time_s']:.3f}s | iters={row['iters']}"
    )


if __name__ == "__main__":
    main()
