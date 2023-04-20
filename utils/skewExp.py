import numpy as np

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def skewExp(s, theta=None):
    if theta is None:
        theta = 1
    
    if s.shape == (3, 1):
        s = skew(s)
    
    n = theta.shape[0] if hasattr(theta, 'shape') else 1
    g = np.zeros((3, 3, n))
    
    for i in range(n):
        g[:, :, i] = np.eye(3) + s * np.sin(theta[i]) + np.linalg.matrix_power(s, 2) * (1 - np.cos(theta[i]))
    
    return g.squeeze()