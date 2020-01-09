import numpy as np
from datetime import datetime
from multiprocessing import Pool
import itertools
from joblib import dump, load
import sys
from parameters import alphas, nmax, N, nd, resfn

# Perceptron function
def perceptron(a, N, n_max):
    """a - alpha
    N - amount of features
    n_max = max number of epochs"""

    P = int(a * N) # Amount of samples
    xi = []
    S = []


    for _ in range(P):
        # Draw N random samples from Gaussian distribution.
        mu, sigma = 0, 0.1
        s = np.random.normal(mu, sigma, N)
        xi.append(s)

        label = 1 if np.random.rand() < 0.5 else -1
        S.append(label)

    w = np.zeros(N) # weights. intially set to 0
    for _ in range(n_max):
        scores = np.zeros(P, dtype=int)

        for mu_t in range(P):
            e_mu_t = np.dot(xi[mu_t], w) * S[mu_t]

            # predict
            scores[mu_t] = 1 if e_mu_t > 0 else 0

            # update weight matrix
            if e_mu_t <= 0:
                w += (1 / N) * xi[mu_t] * S[mu_t]
    
        # compare
        score = np.sum(scores)
        success = score == P

        # print('Epoch {} score [{}/{}]'.format(epoch, score, P))
        if success:
            return 1

    return 0

# Decide between singlecore and multicore execution
multicore = np.size(sys.argv) > 1 and sys.argv[1] == '--multicore'
provider = Pool() if multicore else itertools # Pool() utilises all cores

# Run perceptron
t_start1 = datetime.now()
resvec = list(provider.starmap(perceptron, 
            itertools.product(alphas, np.full(nd, N), [ nmax ]))
        ) # Result vector
t_end1 = datetime.now()

# Dump results to file
resmat = np.reshape(resvec, (np.size(alphas), nd)) # Result matrix
dump(resmat, resfn)

# Runtime report
print('\nTotal execution time')
print('\tt = {}'.format(t_end1 - t_start1))
