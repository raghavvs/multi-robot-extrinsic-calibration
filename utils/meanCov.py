import numpy as np
from scipy.linalg import logm, expm

def vex(s):
    return np.array([-s[1, 2], s[0, 2], -s[0, 1]])

def meanCov(X):
    N = X.shape[2]
    Mean = np.eye(4)
    Cov = np.zeros((6, 6))

    # Initial approximation of Mean
    sum_se = np.zeros((4, 4))
    for i in range(N):
        sum_se = sum_se + logm(X[:, :, i])
    Mean = expm(1 / N * sum_se)

    # Iterative process to calculate the true Mean
    diff_se = np.ones((4, 4))
    max_num = 100
    tol = 1e-5
    count = 1
    while np.linalg.norm(diff_se, 'fro') >= tol and count <= max_num:
        diff_se = np.zeros((4, 4))
        for i in range(N):
            diff_se = diff_se + logm(np.linalg.inv(Mean) @ X[:, :, i])
        Mean = Mean @ expm(1 / N * diff_se)
        count += 1

    # Covariance
    for i in range(N):
        diff_se = logm(np.linalg.inv(Mean) @ X[:, :, i])
        diff_vex = np.hstack((vex(diff_se[:3, :3]), diff_se[:3, 3]))
        Cov = Cov + np.outer(diff_vex, diff_vex)
    Cov = Cov / N

    return Mean, Cov