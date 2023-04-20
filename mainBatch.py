import numpy as np
from solvers import batchSolveXY

def main():
    # Create deterministic input matrices A and B
    A = np.array([[[0.9363, -0.2751, 0.2183, 1.2020],
               [0.2896, 0.9566, -0.0392, -0.1022],
               [-0.1985, 0.0978, 0.9750, 0.3426],
               [0.0, 0.0, 0.0, 1.0]],
              [[0.9938, -0.0975, 0.0599, -0.2246],
               [0.0975, 0.9951, -0.0273, 0.1088],
               [-0.0603, 0.0250, 0.9981, 0.4839],
               [0.0, 0.0, 0.0, 1.0]]])

    B = np.array([[[0.8660, -0.2896, 0.4082, 0.9501],
                [0.5000, 0.8660, -0.0000, -0.5507],
                [-0.0000, 0.0000, 1.0000, 0.5000],
                [0.0, 0.0, 0.0, 1.0]],
                [[0.9603, -0.1944, 0.2014, 0.6231],
                [0.2791, 0.6829, -0.6752, -0.4567],
                [-0.0000, 0.7071, 0.7071, 0.7071],
                [0.0, 0.0, 0.0, 1.0]]])
    
    # Add small perturbations to A and B
    epsilon = 1e-8
    A_perturbed = A + epsilon * np.random.randn(*A.shape)
    B_perturbed = B + epsilon * np.random.randn(*B.shape)

    print("A.shape:", A.shape)
    print("B.shape:", B.shape)
    print("A_perturbed.shape:", A_perturbed.shape)
    print("B_perturbed.shape:", B_perturbed.shape)

    opt = False
    nstd_A = 0.0
    nstd_B = 0.0

    # Call batchSolveXY with perturbed input data
    X, Y, MeanA, MeanB, SigA, SigB = batchSolveXY(A_perturbed, B_perturbed, opt, nstd_A, nstd_B)
    #X, Y, MeanA, MeanB, SigA, SigB = batchSolveXY(A, B, opt, nstd_A, nstd_B)

    # Display results
    print("X:")
    for x in X:
        print(x)
        print()

    print("Y:")
    for y in Y:
        print(y)
        print()

    print("MeanA:")
    print(MeanA)
    print()

    print("MeanB:")
    print(MeanB)
    print()

    print("SigA:")
    print(SigA)
    print()

    print("SigB:")
    print(SigB)
    print()

if __name__ == "__main__":
    main()