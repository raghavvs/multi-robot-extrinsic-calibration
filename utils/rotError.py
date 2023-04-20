import numpy as np
import so3Vec
import skewLog

def rotError(X1, X2):
    err = np.linalg.norm(so3Vec(skewLog(X1[:3, :3].T @ X2[:3, :3])))
    return err