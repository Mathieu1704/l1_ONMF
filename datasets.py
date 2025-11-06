# datasets.py
from __future__ import annotations
import numpy as np
from scipy.io import loadmat
from scipy import sparse

def load_doc_mat(path: str):
    """
    Charge un dataset document .mat (style repo KL-ONMF).
    Attend des clés: 'dtm' (doc-term), 'classid' (labels).
    Le script MATLAB fait dtm = dtm'; on reproduit donc la transposée.
    Retourne X (sparse CSR si possible), y (np.ndarray), r (int).
    """
    mat = loadmat(path, squeeze_me=True)
    if 'dtm' not in mat or 'classid' not in mat:
        raise ValueError(f"{path}: clés 'dtm' et 'classid' nécessaires.")
    dtm = mat['dtm']
    # Transpose comme dans experiment_document.m (dtm = dtm')
    if sparse.issparse(dtm):
        X = dtm.T.tocsr().astype(float)
    else:
        X = np.asarray(dtm, dtype=float).T
    y = np.asarray(mat['classid']).astype(int).ravel()
    r = int(np.max(y))
    return X, y, r

def load_hsi_mat(path: str):
    """
    Charge un dataset HSI .mat (style Moffet/Samson/Jasper).
    Retourne X (m x n dense), r (int si présent, sinon None),
    Wtrue (m x r si présent, sinon None), dimx/dimy si présents.
    """
    mat = loadmat(path, squeeze_me=True)
    keys = mat.keys()
    X = np.asarray(mat['X'], dtype=float) if 'X' in keys else None
    r = int(mat['r']) if 'r' in keys else None
    Wtrue = np.asarray(mat['Wtrue'], dtype=float) if 'Wtrue' in keys else None
    dimx = int(mat['dimx']) if 'dimx' in keys else None
    dimy = int(mat['dimy']) if 'dimy' in keys else None
    if X is None:
        raise ValueError(f"{path}: clé 'X' absente.")
    return X, r, Wtrue, dimx, dimy
