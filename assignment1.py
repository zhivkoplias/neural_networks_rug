import numpy as np

#function
def perceptron(a, N, n_max):
    """a - alpha
    N - amount of features
    P - amount of samples
    n_max = max number of epochs"""
    P = int(a*N) # Amount of samples
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

    w = np.zeros(N)

    for _ in range(n_max):
        scores = np.zeros(P, dtype=int)

        for mu_t in range(P):
            e_mu_t = np.dot(xi[mu_t], w) * S[mu_t]

            # predict
            scores[mu_t] = 1 if e_mu_t > 0 else 0

            # update weight matrix
            if e_mu_t <= 0:
                w = w + a * xi[mu_t] * S[mu_t]
    
        # compare
        score = np.sum(scores)
        success = score == P

        # print('Epoch {} score [{}/{}]'.format(epoch, score, P))
        if success:
            return 1

    return 0

#plot
alphas = np.linspace(0.75, 3, 10)
convergence = np.empty([0, 1])
max_epochs = 100

for alpha in alphas:
    print('alpha = {}'.format(alpha))
    results = []

    for j in range(10):
        result = perceptron(alpha, 20, max_epochs)
        results.append(result)
        convergence = np.append(convergence, (result))

    hits = np.sum(results)
    total = np.size(results)
    print('\t{}/{} randomly generated datasets ended early in {} epochs'.format(
        hits, total, max_epochs))
