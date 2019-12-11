import numpy as np
from datetime import datetime
from multiprocessing import Pool
import itertools
import pickle
import sys

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

        # label = np.random.randint(2)
        label = 1 if np.random.rand() < 0.5 else -1
        S.append(label)

    w = np.zeros(N) # weights
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
alphas = np.linspace(0.75, 3, 10)
nmax = 100      # max number of epochs - before stopping without convergence
N = 10          # amount of features - to use for the dataset
nd = 50         # number of datasets - to generate for each alpha

# Decide between singlecore and multicore execution
multicore = np.size(sys.argv) > 1 and sys.argv[1] == '--multicore'
provider = Pool() if multicore else itertools # Pool() utilises all cores

# Run perceptron
t_start1 = datetime.now()
resultsList = list(provider.starmap(perceptron, 
        itertools.product(alphas, np.full(nd, N), [ nmax ])))
t_end1 = datetime.now()

# Print results
resultsMatrix = np.reshape(resultsList, (np.size(alphas), nd))
print('Results:')
for i, res in enumerate(resultsMatrix):
    a = alphas[i]
    hits = np.sum(res)
    total = np.size(res)

    print('\ta = {}:\t[{}/{}] convergences in {} epochs'.format(
        a, hits, total, nmax))

# Dump results to file
fileObject = open('./nmax={},N={},nd={},a={}.pickle'
    .format(nmax, N, nd, a),'wb')
pickle.dump(resultsMatrix, fileObject)
fileObject.close()

# Runtime report
print('\nTotal execution time')
print('\tt = {}'.format(t_end1 - t_start1))
