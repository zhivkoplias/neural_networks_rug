import numpy as np

# Perceptron parameters
alphas = np.linspace(0.75, 3, 10)
nmax = 100      # max number of epochs - before stopping without convergence
N = 10          # amount of features - to use for the dataset
nd = 50         # number of datasets - to generate for each alpha

# Files
resfn = './nmax={},N={},nd={}.pickle'.format(nmax, N, nd) # results filename