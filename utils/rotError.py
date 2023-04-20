import numpy as np
from utils import so3Vec
from utils import skewLog

def rotError(X1, X2):
    err = np.linalg.norm(so3Vec(skewLog(X1[:3, :3].T @ X2[:3, :3])))
    return err