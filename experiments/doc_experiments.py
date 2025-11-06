# experiments/doc_experiments.py
# ==============================
# L1-ONMF sur les 15 jeux de documents .mat sans argparse (tout est ici).
# Modifie le bloc PARAMÈTRES ci-dessous et lance simplement:
#     python experiments/doc_experiments.py

# ===== PARAMÈTRES =====
DATA_DIR = "./data/docs"           # dossier des .mat (NG20.mat, classic.mat, ...)
OUT_CSV  = "results_docs.csv"      # fichier CSV de sortie
MAXITER  = 100
TOL      = 1e-6
SEED     = 0
# DATASETS = [
#     "NG20.mat","ng3sim.mat","classic.mat","ohscal.mat","k1b.mat","hitech.mat",
#     "reviews.mat","sports.mat","la1.mat","la12.mat","la2.mat","tr11.mat",
#     "tr23.mat","tr41.mat","tr45.mat"
# ]
# Pour un test rapide, commente la liste complète ci-dessus et décommente:
# DATASETS = ["classic.mat", "la1.mat"]
DATASETS = ["classic.mat"]

# ======================

import time, csv
from pathlib import Path
import sys
import numpy as np
from scipy import sparse  # si besoin plus tard

# -- rendre importables les modules locaux (datasets, l1_onmf, metrics, ...) --
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import load_doc_mat
from l1_onmf import alternating_l1_onmf, L1ONMFOptions
from metrics import clustering_accuracy_hungarian, ari, nmi


def run_one(path):
    X, y, r = load_doc_mat(path)
    t0 = time.perf_counter()
    W, H, info = l1_onmf(
        X, r,
        maxiter=MAXITER,
        tol=TOL,
        init_seed=SEED,
        # exemples d’options possibles si ton core les expose :
        # init="snpa",
        # enforce_W_nonneg=False,
        verbose=False,
    )
    t1 = time.perf_counter()

    c_pred = np.asarray(H).argmax(axis=0) + 1  # si labels GT commencent à 1
    return {
        "dataset": Path(path).name,
        "m": X.shape[0],
        "n": X.shape[1],
        "r": r,
        "acc": clustering_accuracy_hungarian(y, c_pred),
        "ari": ari(y, c_pred),
        "nmi": nmi(y, c_pred),
        "time_s": t1 - t0,
        "iters": info.get("n_iter", None),
    }

def main():
    data_dir = Path(DATA_DIR)
    rows = []
    for name in DATASETS:
        p = data_dir / name
        print(f"==> {name}")
        metrics = run_one(str(p))
        rows.append(metrics)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset","m","n","r","acc","ari","nmi","time_s","iters"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Résultats écrits dans {OUT_CSV}")

if __name__ == "__main__":
    main()
