import numpy as np


def split_largest_cluster(H: np.ndarray):
    """
    Utility: find largest cluster (row of H with most nonzeros), return its indices.
    """
    sizes = np.sum(H > 0, axis=1)
    k = int(np.argmax(sizes))
    Kj = np.where(H[k, :] > 0)[0]
    return k, Kj

def ensure_nonempty_clusters(H: np.ndarray):
    """
    Répare H en gardant la contrainte hard-clustering:
    - 1 seul non-zéro par colonne
    - pas de cluster vide
    """
    H = np.asarray(H, dtype=float).copy()
    k, n = H.shape

    # assignment courant (hard)
    assign = np.argmax(H, axis=0)
    sizes = np.bincount(assign, minlength=k)
    empties = np.where(sizes == 0)[0]

    for i in empties:
        donor = int(np.argmax(sizes))
        donor_cols = np.where(assign == donor)[0]

        # on prend la colonne la "moins forte" du donor (scale la plus petite)
        j = donor_cols[np.argmin(H[donor, donor_cols])]

        # MOVE colonne j vers le cluster vide i
        H[:, j] = 0.0
        H[i, j] = 1.0  # valeur arbitraire > 0 ; la normalisation derrière fixera l’échelle

        assign[j] = i
        sizes[i] += 1
        sizes[donor] -= 1

    return H
