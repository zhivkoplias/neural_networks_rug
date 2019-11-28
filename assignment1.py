import numpy as np

a = 1 # alpha
N = 20 # Amount of features
P = a*N # Amount of samples

xi = []
S = []
for i in range(P):
    # Draw N random samples from Gaussian distribution.
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, N)
    xi.append(s)

    label = np.random.randint(2)
    S.append(label)

w = np.zeros(N)

n_max = 100
for epoch in range(n_max):
    S_pred = np.zeros(P, dtype=int)

    for mu_t in range(P):
        e_mu_t = np.dot(xi[mu_t], w) * S[mu_t]

        # predict
        S_pred[mu_t] = 1 if e_mu_t > 0 else 0

        # update weight matrix
        if e_mu_t <= 0:
            w = w + a * xi[mu_t] * S[mu_t]
    
    # compare
    correctness = np.equal(S_pred, S)
    score = np.sum(correctness)
    success = np.all(correctness)
    
    print('Epoch {} score [{}/{}]'.format(epoch, score, P))
    if success:
        break

print('End')