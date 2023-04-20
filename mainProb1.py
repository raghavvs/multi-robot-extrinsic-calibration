import numpy as np
from solvers import axbyczProb1

def mainProb1():
    # Create deterministic input matrices
    num_matrices = 2
    A1 = np.zeros((num_matrices, 4, 4))
    B1 = np.zeros((num_matrices, 4, 4))
    C1 = np.zeros((num_matrices, 4, 4))
    A2 = np.zeros((num_matrices, 4, 4))
    B2 = np.zeros((num_matrices, 4, 4))
    C2 = np.zeros((num_matrices, 4, 4))
    
    # Fill in the input matrices with specific examples
    A1[0] = np.array([[1, 2, 3, 1],
                      [0, 1, 0, 2],
                      [0, 0, 1, 3],
                      [0, 0, 0, 1]])
    A1[1] = np.array([[1, 1, 3, 2],
                      [0, 1, 0, 3],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1]])

    B1 = A1.copy()
    C1 = A1.copy()
    A2 = A1.copy()
    B2 = A1.copy()
    C2 = A1.copy()

    # Set opt, nstd1, and nstd2
    opt = True
    nstd1 = 0.01
    nstd2 = 0.01

    # Call the axbyczProb1 function
    X_final, Y_final, Z_final = axbyczProb1(A1, B1, C1, A2, B2, C2, opt, nstd1, nstd2)

    # Display results
    print("X_final:")
    print(X_final)
    print("Y_final:")
    print(Y_final)
    print("Z_final:")
    print(Z_final)

def axbyczProb1(A1, B1, C1, A2, B2, C2, opt, nstd1, nstd2):
    # Implement your axbyczProb1 function here.
    X_final = np.eye(4)
    Y_final = np.eye(4)
    Z_final = np.eye(4)
    return X_final, Y_final, Z_final

if __name__ == "__main__":
    mainProb1()
