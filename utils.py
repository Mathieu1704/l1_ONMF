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

def handle_empty_clusters(X, W, H, assign):
    """
    Transpose de la logique MATLAB 'Deal with empty clusters'.
    Si un cluster est vide, on coupe le plus gros cluster en deux.
    """
    k = W.shape[1]
    
    # Compter la taille des clusters
    cluster_sizes = np.bincount(assign, minlength=k)
    empty_clusters = np.where(cluster_sizes == 0)[0]
    
    if len(empty_clusters) == 0:
        return W, H, assign

    # Pour chaque cluster vide
    for empty_idx in empty_clusters:
        # Trouver le cluster le plus gros (Donor)
        donor_idx = np.argmax(cluster_sizes)
        donor_docs = np.where(assign == donor_idx)[0]
        
        if len(donor_docs) < 2:
            continue # Pas assez de points pour spliter
            
        # Stratégie simple de split (inspirée du code prof):
        # On perturbe légèrement le centroïde du donneur pour créer celui du vide
        W[:, empty_idx] = W[:, donor_idx] * (0.9 + 0.2 * np.random.rand(W.shape[0]))
        
        # On réassigne aléatoirement la moitié des docs du donneur vers le nouveau
        split_mask = np.random.rand(len(donor_docs)) > 0.5
        docs_to_move = donor_docs[split_mask]
        
        assign[docs_to_move] = empty_idx
        
        # Mise à jour des tailles pour la prochaine itération de la boucle
        cluster_sizes[donor_idx] -= len(docs_to_move)
        cluster_sizes[empty_idx] += len(docs_to_move)
        
        # Reset H pour ces clusters (sera recalculé proprement à la prochaine update_H)
        H[empty_idx, docs_to_move] = 1.0
        H[donor_idx, docs_to_move] = 0.0

    return W, H, assign