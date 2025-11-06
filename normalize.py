import numpy as np

def normalize_rows_H_and_rescale_W(H: np.ndarray, W: np.ndarray, eps: float = 1e-16):
    """
    For each row i: alpha_i = ||H[i]||_2; set H[i] /= alpha_i, W[:, i] *= alpha_i.
    This preserves WH and enforces HH^T = I_k when rows are orthogonal.
    """
    H = np.asarray(H, dtype=float)
    W = np.asarray(W, dtype=float)
    row_norms = np.linalg.norm(H, axis=1) + eps
    H_scaled = H / row_norms[:, None]
    W_scaled = W * row_norms[None, :]
    return H_scaled, W_scaled
