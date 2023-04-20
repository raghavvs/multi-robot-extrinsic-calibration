import numpy as np
from solvers import batchSolveXY
from utils import rotError, tranError

def axbyczProb1(A1, B1, C1, A2, B2, C2, opt, nstd1, nstd2):
    A1_fixed = A1[:, :, 0]
    C2_fixed = C2[:, :, 0]

    Z_g, _, MeanC1, MeanB1, _, _ = batchSolveXY(C1, B1, opt, nstd1, nstd2)

    Z_index = 1
    Z = []

    for i in range(Z_g.shape[2]):
        if np.linalg.det(Z_g[:, :, i]) > 0:
            Z.append(Z_g[:, :, i])
            Z_index += 1

    Z = np.stack(Z, axis=2)
    s_Z = Z.shape[2]

    Num = A1.shape[2]
    print("Num:", Num)
    A2_inv = np.zeros((4, 4, Num))
    B2_inv = np.zeros((4, 4, Num))

    print("A2 shape:", A2.shape)
    print("B2 shape:", B2.shape)

    for i in range(Num):
        A2_inv[:, :, i] = np.linalg.inv(A2[:, :, i])
        B2_inv[:, :, i] = np.linalg.inv(B2[:, :, i])

    X_g, _, MeanA2, _, _, _ = batchSolveXY(A2, B2_inv, opt, nstd1, nstd2)
    _, _, _, MeanB2, _, _ = batchSolveXY(A2_inv, B2, opt, nstd1, nstd2)

    X_index = 1
    X = []

    for i in range(X_g.shape[2]):
        if np.linalg.det(X_g[:, :, i]) > 0:
            X.append(X_g[:, :, i])
            X_index += 1

    X = np.stack(X, axis=2)
    s_X = X.shape[2]

    Y = np.zeros((4, 4, 2 * s_X * s_Z))

    for i in range(s_X):
        for j in range(s_Z):
            Y[:, :, (i * s_Z) + j] = (A1_fixed @ X[:, :, i] @ MeanB1 / Z[:, :, j]) / MeanC1
            Y[:, :, (i * s_Z) + j + (s_X * s_Z)] = (MeanA2 @ X[:, :, i] @ MeanB2 / Z[:, :, j]) / C2_fixed_fixed

    s_Y = Y.shape[2]

    cost = np.zeros((s_X, s_Y * s_Z))
    weight = 1.5

    for i in range(s_X):
        for j in range(s_Z):
            for m in range(s_Y):
                left1 = A1_fixed @ X[:, :, i] @ MeanB1
                right1 = Y[:, :, m] @ MeanC1 @ Z[:, :, j]
                diff1 = rotError(left1, right1) + weight * tranError(left1, right1)

                left2 = MeanA2 @ X[:, :, i] @ MeanB2
                right2 = Y[:, :, m] @ C2_fixed @ Z[:, :, j]
                diff2 = rotError(left2, right2) + weight * tranError(left2, right2)

                cost[i, (j * s_Y) + m] = np.linalg.norm(diff1) + np.linalg.norm(diff2)

    I1 = np.argmin(cost)
    I_row, I_col = np.unravel_index(I1, cost.shape)

    X_final = X[:, :, I_row]  # final X

    if I_col % s_Y > 0:
        index_Z = I_col // s_Y + 1
    else:
        index_Z = I_col // s_Y

    Z_final = Z[:, :, index_Z]  # final Z

    if I_col % s_Y > 0:
        index_Y = I_col % s_Y
    else:
        index_Y = s_Y

    Y_final = Y[:, :, index_Y]  # final Y

    return X_final, Y_final, Z_final