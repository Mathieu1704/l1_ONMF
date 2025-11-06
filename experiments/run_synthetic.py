# examples/run_synthetic.py
import numpy as np
from l1_onmf import l1_onmf
from metrics import clustering_accuracy_hungarian

rng = np.random.default_rng(0)
m, n, r = 10, 300, 3

# Génère 3 colonnes "centroïdes" et des échelles positives
W_true = np.abs(rng.normal(size=(m, r)))
H_true = np.zeros((r, n))
y_true = np.zeros(n, dtype=int)
for j in range(n):
    c = rng.integers(0, r)
    s = np.abs(rng.normal(loc=1.0, scale=0.2))
    H_true[c, j] = s
    y_true[j] = c + 1  # labels 1..r

X = W_true @ H_true + 0.05 * rng.normal(size=(m, n))  # petit bruit

W, H, info = l1_onmf(X, r, maxiter=100, tol=1e-6, init_seed=0, verbose=True)

y_pred = H.argmax(axis=0) + 1
acc = clustering_accuracy_hungarian(y_true, y_pred)
print(f"Accuracy synthétique = {acc*100:.2f}% — iters={info.get('n_iter')}")
