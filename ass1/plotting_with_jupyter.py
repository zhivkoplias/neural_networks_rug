#plot for basic part

nmax = 100      # max number of epochs - before stopping without convergence
N = 20          # amount of features - to use for the dataset
nd = 50     # number of datasets - to generate for each alpha

alphas = np.linspace(0.75, 3, 30)

alphas = np.linspace(0.75, 3, 10)

convergence_av_basic = np.empty([0,10])

for i in alphas:
    convergence = np.empty([0, 1])
    for j in range(nd):
        convergence = np.append(convergence,(perceptron(i, N, nmax)))
    convergence_av_basic = np.append(convergence_av_basic,np.mean(convergence))

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 24})
plt.plot(alphas, convergence_av_basic, marker='o', linestyle='-', color='b')
plt.xlabel('alpha')
plt.show()
