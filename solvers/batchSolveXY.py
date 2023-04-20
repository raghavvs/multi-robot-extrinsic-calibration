import numpy as np
from utils import meanCov, so3Vec

def batchSolveXY(A, B, opt, nstd_A=None, nstd_B=None):
    X_candidate = np.zeros((4, 4, 8))
    Y_candidate = np.zeros((4, 4, 8))

    MeanA, SigA = meanCov(A)
    MeanB, SigB = meanCov(B)

    if opt:
        SigA = SigA - nstd_A * np.eye(6, 6)
        SigB = SigB - nstd_B * np.eye(6, 6)

    VA, _ = np.linalg.eig(SigA[:3, :3])
    VB, _ = np.linalg.eig(SigB[:3, :3])

    Q1 = np.eye(3)
    Q2 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    Q3 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    Q4 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    Rx_solved = np.zeros((3, 3, 8))

    Rx_solved[:, :, 0] = VA @ Q1 @ VB.T
    Rx_solved[:, :, 1] = VA @ Q2 @ VB.T
    Rx_solved[:, :, 2] = VA @ Q3 @ VB.T
    Rx_solved[:, :, 3] = VA @ Q4 @ VB.T
    Rx_solved[:, :, 4] = VA @ (-Q1) @ VB.T
    Rx_solved[:, :, 5] = VA @ (-Q2) @ VB.T
    Rx_solved[:, :, 6] = VA @ (-Q3) @ VB.T
    Rx_solved[:, :, 7] = VA @ (-Q4) @ VB.T

    for i in range(8):
        matrix = Rx_solved[:, :, i].T @ SigA[:3, :3] @ Rx_solved[:, :, i]
        det = np.linalg.det(matrix)
        if np.isclose(det, 0):
            print(f"Matrix at index {i} is singular with determinant {det}")
        else:
            tx_temp = so3Vec((np.linalg.inv(matrix) @
                            (SigB[:3, 3:6] - Rx_solved[:, :, i].T @ SigA[:3, 3:6] @ Rx_solved[:, :, i])).T)
            tx = -Rx_solved[:, :, i] @ tx_temp
            X_candidate[:, :, i] = np.block([[Rx_solved[:, :, i], tx[:, None]], [[0, 0, 0, 1]]])
            Y_candidate[:, :, i] = MeanA @ X_candidate[:, :, i] / (MeanB)

    return X_candidate, Y_candidate, MeanA, MeanB, SigA, SigB