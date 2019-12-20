from parameters import alphas, nmax, resfn, nd
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Read results file
resmat = load(resfn)

yvals = np.sum(resmat, axis=1) / nd
data = pd.DataFrame({'x': alphas, 'y': yvals})
sns.lineplot(data=data)
plt.xlabel('alphas')
plt.ylabel('fraction of convergences')
plt.show()
