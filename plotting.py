from parameters import alphas, nmax, resultsfn
import pickle
import numpy as np

# Read results file
resfile = open(resultsfn, 'r')
resmat = pickle.load(resfile)

# Create a plot displaying the results.
for i, res in enumerate(resmat):
    print('\t[alpha = {:.2f}] {:02d}/{:02d} datasets converged in {} epochs'
        .format(alphas[i], np.sum(res), np.size(res), nmax))