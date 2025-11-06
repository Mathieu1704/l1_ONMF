import numpy as np

def rel_l1_error(X: np.ndarray, W: np.ndarray, H: np.ndarray, eps: float = 1e-16) -> float:
    num = np.sum(np.abs(X - W @ H))
    den = np.sum(np.abs(X)) + eps
    return float(num / den)
