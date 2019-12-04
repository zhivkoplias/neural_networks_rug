import numpy as np
from datetime import datetime
from multiprocessing import Pool
from itertools import product

t_start1 = datetime.now()
# t_generate = t_start      # time spent generating sample data
# t_perceptron = t_start    # time spent executing the perceptron

#function
def perceptron(a, N, n_max):
    """a - alpha
    N - amount of features
    n_max = max number of epochs"""
    P = int(a*N) # Amount of samples
    xi = []
    S = []

    print('perceptron(a={}, N={}, n_max={})'.format(a, N, n_max))
    
    # global t_generate
    # global t_perceptron
    # timing_1 = datetime.now()

    for _ in range(P):
        # Draw N random samples from Gaussian distribution.
        mu, sigma = 0, 0.1
        s = np.random.normal(mu, sigma, N)
        xi.append(s)

        # label = np.random.randint(2)
        label = 1 if np.random.rand() < 0.5 else -1
        S.append(label)

    # timing_2 = datetime.now()
    # t_generate += timing_2 - timing_1

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
            # t_perceptron += datetime.now() - timing_2
            return 1

    # t_perceptron += datetime.now() - timing_2
    return 0

# Execution parameters
alphas = np.linspace(0.75, 3, 10)
convergence = np.empty([0, 1])
max_epochs = 100
number_sets = 50
N = 10

# Create multiprocessing pool
with Pool() as pool: # by default, uses n = os.cpu_count()
    # mock results- remove in production
    results = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # (!) multiprocessing module does not work in VSCode debugging
    results = pool.starmap(perceptron, 
            product(alphas, np.full(number_sets, N), [ max_epochs ])
    )

    print(results)

t_end1 = datetime.now()

# Using for loops.
t_start2 = datetime.now()

for alpha in alphas:
    print('alpha = {}'.format(alpha))
    results = []
    t_alpha_start = datetime.now()

    for _ in range(number_sets):
        result = perceptron(alpha, N, max_epochs)
        results.append(result)
        convergence = np.append(convergence, (result))

    # timing
    t_end = datetime.now()
    print('\tExecution time = {}'.format(t_end - t_alpha_start))
    # results
    hits = np.sum(results)
    total = np.size(results)
    print('\t{}/{} randomly generated datasets ended early in {} epochs'.format(
        hits, total, max_epochs))
# print('\tTime generating datasets = {}'.format(t_generate - t_start))
# print('\tTime executing perceptron = {}'.format(t_perceptron - t_start))


t_end2 = datetime.now()






# Runtime report
print('\nTotal execution time')
print('\tUsing multiprocessing = {}'.format(t_end1 - t_start1))
print('\tUsing for loops = {}'.format(t_end2 - t_start2))
