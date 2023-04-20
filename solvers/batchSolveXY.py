import numpy as np
from utils import meanCov, so3Vec

def batchSolveXY(A, B, opt, nstd_A, nstd_B):
    X_candidate = np.zeros((4, 4, 8))
    Y_candidate = np.zeros((4, 4, 8))

    MeanA, SigA = meanCov(A)
    MeanB, SigB = meanCov(B)

    if opt:
        SigA = SigA - nstd_A * np.eye(6, 6)
        SigB = SigB - nstd_B * np.eye(6, 6)

    eigenvalues_A, VA = np.linalg.eig(SigA[:3, :3])
    eigenvalues_B, VB = np.linalg.eig(SigB[:3, :3])

    # Sort the eigenvalues and eigenvectors based on the eigenvalues in descending order
    sorted_indices_A = np.argsort(eigenvalues_A)[::-1]
    VA = VA[:, sorted_indices_A]

    sorted_indices_B = np.argsort(eigenvalues_B)[::-1]
    VB = VB[:, sorted_indices_B]

    # Verify eigenvectors for VA
    eigenvalues_A, eigenvectors_A = np.linalg.eig(SigA[:3, :3])
    for i in range(3):
        v = eigenvectors_A[:, i]
        Av = SigA[:3, :3] @ v
        lv = eigenvalues_A[i] * v
        assert np.allclose(Av, lv), f"Failed for VA: eigenvector {i+1}"

    # Verify eigenvectors for VB
    eigenvalues_B, eigenvectors_B = np.linalg.eig(SigB[:3, :3])
    for i in range(3):
        v = eigenvectors_B[:, i]
        Bv = SigB[:3, :3] @ v
        lv = eigenvalues_B[i] * v
        assert np.allclose(Bv, lv), f"Failed for VB: eigenvector {i+1}"


    print("VA: ", VA)
    print("VB: ", VB)

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

    X = np.zeros((4, 4, 8))
    Y = np.zeros((4, 4, 8))
    X_candidate = np.zeros((4, 4, 8))
    Y_candidate = np.zeros((4, 4, 8))

    for i in range(8):
        temp = np.linalg.inv(Rx_solved[:,:,i].T @ SigA[:3, :3] @ Rx_solved[:,:,i]) @ \
                (SigB[:3, 3:6] - Rx_solved[:,:,i].T @ SigA[:3, 3:6] @ Rx_solved[:,:,i]).T

        tx = -Rx_solved[:,:,i] @ so3Vec(temp)

        X_candidate[:, :, i] = np.vstack((np.hstack((Rx_solved[:, :, i], tx[:, np.newaxis])), [0, 0, 0, 1]))
        Y_candidate[:, :, i] = MeanA @ X_candidate[:, :, i] @ np.linalg.inv(MeanB)

        # Set the output X and Y
        X[:, :, i] = X_candidate[:, :, i]
        Y[:, :, i] = Y_candidate[:, :, i]

    return X, Y, MeanA, MeanB, SigA, SigB