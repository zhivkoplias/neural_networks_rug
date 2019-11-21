import numpy as np

a = 1 # <to be implememented>
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

w = 0

print('End')