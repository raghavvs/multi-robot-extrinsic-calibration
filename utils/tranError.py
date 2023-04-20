import numpy as np

def tranError(X1, X2):
    error = np.linalg.norm(X1[:3, 3] - X2[:3, 3])
    return error