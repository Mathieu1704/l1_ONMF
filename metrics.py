# metrics.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def clustering_accuracy_hungarian(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Aligne y_pred sur y_true via Hungarian et renvoie l'accuracy (0..1).
    y_true: shape (n,), entiers [1..r] ou [0..r-1]
    y_pred: shape (n,), entiers [1..r] ou [0..r-1]
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # normaliser labels en [0..r-1]
    y_true_u, y_true_inv = np.unique(y_true, return_inverse=True)
    y_pred_u, y_pred_inv = np.unique(y_pred, return_inverse=True)
    r = max(len(y_true_u), len(y_pred_u))
    # matrice de coût = -compte des co-occurrences
    C = np.zeros((r, r), dtype=np.int64)
    for t, p in zip(y_true_inv, y_pred_inv):
        C[t, p] += 1
    # on maximise C -> on minimise -C
    row_ind, col_ind = linear_sum_assignment(-C)
    correct = C[row_ind, col_ind].sum()
    return correct / len(y_true)

def mrsa(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean Removed Spectral Angle (en pourcents, 0..100).
    A et B: matrices m x r (colonnes alignées par permutation au préalable).
    """
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if A.shape != B.shape:
        raise ValueError("A et B doivent avoir les mêmes dimensions (m x r).")
    r = A.shape[1]
    scores = []
    for k in range(r):
        x = A[:, k] - A[:, k].mean()
        y = B[:, k] - B[:, k].mean()
        nx = np.linalg.norm(x) + 1e-16
        ny = np.linalg.norm(y) + 1e-16
        s = np.clip(np.dot(x, y) / (nx * ny), -1.0, 1.0)
        angle = np.arccos(s) * 100.0 / np.pi
        scores.append(abs(angle))
    return float(np.mean(scores))

def reorder_by_hungarian(W_ref: np.ndarray, W_est: np.ndarray):
    """
    Trouve la meilleure permutation de colonnes de W_est pour s'aligner sur W_ref (via coût MRSA pairwise).
    Retourne W_est permuté et l’assignation (liste d'indices).
    """
    r = W_ref.shape[1]
    # matrice de distances (MRSA)
    D = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            D[i, j] = mrsa(W_ref[:, [i]], W_est[:, [j]])
    row_ind, col_ind = linear_sum_assignment(D)
    return W_est[:, col_ind], col_ind

def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return adjusted_rand_score(y_true, y_pred)

def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return normalized_mutual_info_score(y_true, y_pred)
