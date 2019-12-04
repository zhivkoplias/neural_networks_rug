#function
def perceptron(a, N, n_max):
    """a - alpha
    N - amount of features
    P - amount of samples
    n_max = max number of epochs"""
    P = int(a*N) # Amount of samples

    for i in range(P):
        # Draw N random samples from Gaussian distribution.
        mu, sigma = 0, 0.1
        s = np.random.normal(mu, sigma, N)
        xi.append(s)

    # label = np.random.randint(2)
    label = 1 if np.random.rand() < 0.5 else -1
    S.append(label)

    w = np.zeros(N)

    for epoch in range(n_max):
        # S_pred = np.zeros(P, dtype=int)
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

   #     print('Epoch {} score [{}/{}]'.format(epoch, score, P))
        if success:
            return 1
        if epoch >= n_max-1:
            return 0

#plot
alphas = np.linspace(0.75, 3, 10)
convergence = np.empty([0, 1])

for i in alphas:
    for j in range(10):
        print(i)
        convergence = np.append(convergence,(perceptron(i, 20, 100)))

    print(convergence)
