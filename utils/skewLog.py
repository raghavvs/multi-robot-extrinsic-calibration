import numpy as np

def isequalf(A, B, tol=1e-6):
    return np.allclose(A, B, rtol=0, atol=tol)

def skewLog(R):
    if isequalf(R, np.eye(3), 1e-6):
        w_hat = np.zeros((3, 3))
    else:
        val = (np.trace(R) - 1) / 2
        if val > 1:
            val = 1
        elif val < -1:
            val = -1
        theta = np.arccos(val)
        if 0 == theta:
            w_hat = np.zeros((3, 3))
        elif abs(np.pi - theta) < 1e-6:
            M = (R - np.eye(3)) / 2
            m1 = M[0, 0]
            m2 = M[1, 1]
            m3 = M[2, 2]
            w_hat = theta * np.array([[0, -np.sqrt((m3 - m1 - m2) / 2), np.sqrt((m2 - m1 - m3) / 2)],
                                      [np.sqrt((m3 - m1 - m2) / 2), 0, -np.sqrt((m1 - m2 - m3) / 2)],
                                      [-np.sqrt((m2 - m1 - m3) / 2), np.sqrt((m1 - m2 - m3) / 2), 0]])
        else:
            w_hat = (R - R.T) / (2 * np.sin(theta)) * theta
    return w_hat