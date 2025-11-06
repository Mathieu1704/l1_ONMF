import numpy as np

def split_largest_cluster(H: np.ndarray):
    """
    Utility: find largest cluster (row of H with most nonzeros), return its indices.
    """
    sizes = np.sum(H > 0, axis=1)
    k = int(np.argmax(sizes))
    Kj = np.where(H[k, :] > 0)[0]
    return k, Kj

def ensure_nonempty_clusters(H: np.ndarray, X: np.ndarray):
    """
    If a row i of H is all zeros, reassign a few worst-reconstructed points to that cluster.
    (Simple heuristic; can be replaced by a proper split or reinit.)
    """
    k, n = H.shape
    empties = np.where(np.sum(H > 0, axis=1) == 0)[0]
    for i in empties:
        # pick a random column to start the cluster
        j = np.random.randint(0, n)
        H[i, j] = np.linalg.norm(X[:, j])  # a rough positive scale
    return H
