# experiments/hsi_experiments.py
# ==============================
# L1-ONMF sur un jeu hyperspectral .mat sans argparse.
# Modifie le bloc PARAMÈTRES ci-dessous et lance simplement:
#     python experiments/hsi_experiments.py

# ===== PARAMÈTRES =====
DATA_DIR = "./data/hsi"  # dossier des .mat HSI
EXP      = "Moffet"      # "Moffet" | "Samson" | "Jasper" (ignoré si MAT_PATH non vide)
MAT_PATH = ""            # ex: "./data/hsi/Moffet.mat"
MAXITER  = 100
TOL      = 1e-6
SEED     = 0
# ======================

import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "hsi"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from datasets import load_hsi_mat
from l1_onmf import alternating_l1_onmf, L1ONMFOptions
from metrics import reorder_by_hungarian, mrsa

def pick_path():
    if MAT_PATH:
        return MAT_PATH
    name = {"Moffet":"Moffet.mat", "Samson":"Samson.mat", "Jasper":"Jasper.mat"}[EXP]
    return str(DATA_DIR / name)


def main():
    mat_path = pick_path()
    print(f"==> {mat_path}")
    X, r, Wtrue, dimx, dimy = load_hsi_mat(mat_path)
    print(f"X={X.shape}, r={r}")

    # Construire les options pour l'algo
    opts = L1ONMFOptions(
        r=r,
        maxiter=MAXITER,
        delta=TOL,
        seed=SEED,
        verbose=False,
        log_errors=False,  # passe à True si tu veux voir la convergence
    )

    t0 = time.perf_counter()
    W, H, info = alternating_l1_onmf(X, opts)
    t1 = time.perf_counter()
    print(f"iters={info.get('num_iter', None)}, time={t1 - t0:.2f}s")

    if Wtrue is not None:
        W_aligned, _ = reorder_by_hungarian(Wtrue, W)
        score = mrsa(Wtrue, W_aligned)
        print(f"MRSA (plus petit est meilleur) = {score:.2f}%")
    else:
        print("Wtrue non fourni dans le .mat — MRSA non évalué.")

if __name__ == "__main__":
    main()
