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

    # debugging
    # print('perceptron(a={}, N={}, n_max={})'.format(a, N, n_max))

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

    w = np.zeros(N) # weights. intiially set to 0
    for _ in range(n_max):
        scores = np.zeros(P, dtype=int)

        for mu_t in range(P):
            e_mu_t = np.dot(xi[mu_t], w) * S[mu_t]

            # predict
            scores[mu_t] = 1 if e_mu_t > 0 else 0

            # update weight matrix
            if e_mu_t <= 0:
                w += a * xi[mu_t] * S[mu_t]
    
        # compare
        score = np.sum(scores)
        success = score == P

        # print('Epoch {} score [{}/{}]'.format(epoch, score, P))
        if success:
            return 1

    return 0

# Perceptron parameters
# alphas = np.linspace(0.75, 3, 10)
# nmax = 100      # max number of epochs - before stopping without convergence
# N = 10          # amount of features - to use for the dataset
# nd = 50         # number of datasets - to generate for each alpha

# Decide between singlecore and multicore execution
multicore = np.size(sys.argv) > 1 and sys.argv[1] == '--multicore'
provider = Pool() if multicore else itertools # Pool() utilises all cores

# Run perceptron
t_start1 = datetime.now()
resvec = list(provider.starmap(perceptron, 
        itertools.product(alphas, np.full(nd, N), [ nmax ]))) # Result vector
t_end1 = datetime.now()

# Print results
resmat = np.reshape(resvec, (np.size(alphas), nd)) # Result matrix
print('Results:')
for i, res in enumerate(resmat):
    print('\t[alpha = {:.2f}] {:02d}/{:02d} datasets converged in {} epochs'
        .format(alphas[i], np.sum(res), np.size(res), nmax))

# Dump results to file
dump(resmat, resfn)

# Runtime report
print('\nTotal execution time')
print('\tt = {}'.format(t_end1 - t_start1))
