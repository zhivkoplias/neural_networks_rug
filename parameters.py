import numpy as np

# Perceptron parameters
aincr = 200     # alpha increment level
nmax = 100      # max number of epochs - before stopping without convergence
N = 1000        # amount of features - to use for the dataset
nd = 50         # number of datasets - to generate for each alpha

# Spawn values of alpha
alphas = np.linspace(0.75, 3, aincr)

# Files
resfn = './nmax={},N={},nd={},aincr={}.pickle'
    .format(nmax, N, nd, aincr) # results filename