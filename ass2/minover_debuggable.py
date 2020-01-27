#import libraries
import numpy as np
import math
from operator import itemgetter
import random
from matplotlib import pyplot as plt

#params
nd = 10 #num of trials
nmax = 100 #num of epochs
N = 20 #num of features
alphas = np.linspace(0.1, 10, 10) #alpha

av_errors = [] #av gen errors
for alpha in alphas:
    P = int(alpha * N) #num of samples
    sum_errors = [] #gen error calculation

    for j in range(nd):

        #prepare all vectors
        mu, sigma = 0, 1
        xi = np.random.normal(mu, sigma, [P, N]) #initial values
        w_t = np.random.normal(mu, sigma, N) #initial weights, teacher weights, have to be random not just '1'
        w_t = w_t * np.sqrt(N) / np.linalg.norm(w_t,2)
        S = np.sign(np.dot(xi, w_t)) #initial labels (teacher weights),
        #they are the products of weights and xi vector
        w = np.zeros(N) #learned weights (students weights)

        converg = 0 #check convergence
        initial_lowest_stab = 0 #initial κν; t=0

        for i in range(nmax): #actual algorithm starts here

            data = []

            #calculate κν(t)
            for mu_t in range(P):
                current_example = xi[mu_t]
                e_mu_t = np.dot(np.dot(current_example, w), S[mu_t])
                data.append([e_mu_t, xi[mu_t], S[mu_t]])

            #determine the minimal stability
            data = sorted(data, key=itemgetter(0), reverse=False)
            lowest_stab = data[0][0]

            #determine the sample with minimal stability and its label
            min_data = data[0][1]
            min_label = data[0][2]

            #hebbian update
            w = w+(min_data*min_label)/N

            if lowest_stab >= initial_lowest_stab: #if the algorithm found the larger distance
                initial_lowest_stab = lowest_stab
                converge = 0 #then the algorithm converges
            else: #if the distance with the new example that the algorithm found is larger the one that we already have
                converge = converge+1 #then increase counter by 1
                if converge >= P: #once it reaches threshold (larger than P as it is stated in the assignment)
                    break #stop the algorithm

        #calculate the error
        error = 1/np.pi * math.acos((np.dot(w, w_t))/(np.linalg.norm(w,2)*np.linalg.norm(w_t,2)))
        sum_errors.append(error)

    av_error = sum(sum_errors)/len(sum_errors)
    print('alpha:')
    print(alpha)
    print('av error:')
    print(av_error)
    print('\n')
    av_errors.append(av_error)

plt.plot(av_errors)
plt.ylabel('Generalization error')
plt.xlabel('alpha')
plt.title('Generalization error Minover perceptron')
plt.show()