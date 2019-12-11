from parameters import alphas, nmax, resfn
from joblib import load
import numpy as np

# Read results file
resmat = load(resfn)

# Create a plot displaying the results.
for i, res in enumerate(resmat):
    print('\t[alpha = {:.2f}] {:02d}/{:02d} datasets converged in {} epochs'
        .format(alphas[i], np.sum(res), np.size(res), nmax))