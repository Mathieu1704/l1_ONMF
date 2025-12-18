import numpy as np
<<<<<<< Updated upstream
=======
import scipy.sparse as sp
>>>>>>> Stashed changes

def split_largest_cluster(H: np.ndarray):
    """
    Utility: find largest cluster (row of H with most nonzeros), return its indices.
    """
    sizes = np.sum(H > 0, axis=1)
    k = int(np.argmax(sizes))
    Kj = np.where(H[k, :] > 0)[0]
    return k, Kj

def ensure_nonempty_clusters(H: np.ndarray, X, eps: float = 1e-12):
    """
    Ensure each cluster (row of H) has at least one assigned column.
    If a cluster i is empty, reassign one column j (currently in the largest cluster)
    to cluster i, with a positive scale.

    Works for both dense numpy arrays and scipy sparse matrices (CSC/CSR).
    """
    H = np.asarray(H, dtype=float)
    k, n = H.shape

    # current assignment
    assign = np.argmax(H, axis=0)
    counts = np.bincount(assign, minlength=k)

    # helper: compute ||X[:, j]||_2 efficiently
    def col_norm2(j: int) -> float:
        if sp is not None and sp.isspmatrix(X):
            # Ensure CSC for fast column slicing
            Xc = X.tocsc()
            start, end = Xc.indptr[j], Xc.indptr[j + 1]
            data = Xc.data[start:end]
            return float(np.sqrt(np.dot(data, data))) if data.size else 0.0
        else:
            xj = np.asarray(X[:, j], dtype=float).ravel()
            return float(np.linalg.norm(xj))

    # While there is an empty cluster
    empties = np.where(counts == 0)[0]
    if empties.size == 0:
        return H

    for i in empties:
        # pick a column from the currently largest cluster to steal
        donor_cluster = int(np.argmax(counts))
        donor_cols = np.where(assign == donor_cluster)[0]
        if donor_cols.size == 0:
            # fallback: pick any column
            j = 0
        else:
            j = int(donor_cols[0])

        # reassign column j to empty cluster i
        H[:, j] = 0.0
        scale = col_norm2(j)
        if scale <= eps:
            scale = 1.0
        H[i, j] = scale

        # update bookkeeping
        assign[j] = i
        counts[donor_cluster] -= 1
        counts[i] += 1

    return H
