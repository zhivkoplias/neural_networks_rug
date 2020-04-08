from parameters import alphas, nmax, resfn, nd
from joblib import load
import numpy as np
import matplotlib.pyplot as plt

# Read results file
resmat = load(resfn)
yvals = np.sum(resmat, axis=1) / nd

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 24})
plt.plot(alphas, yvals, marker='o', linestyle='-', color='b')
plt.xlabel('alpha')
plt.ylabel('Q l.s.')
plt.show()
