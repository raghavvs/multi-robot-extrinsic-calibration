import numpy as np

def so3Vec(X):
    if X.shape[1] == 3:                                 # If input is skew-sym change to vector
        g = np.array([-X[1, 2], X[0, 2], -X[0, 1]])
    else:                                               # If input is vector change to skew-sym
        g = np.array([[0, -X[2], X[1]],
                      [X[2], 0, -X[0]],
                      [-X[1], X[0], 0]])
    return g