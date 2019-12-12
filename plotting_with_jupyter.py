alphas = np.linspace(0.75, 3, 10)

convergence_av_basic = np.empty([0,10])

for i in alphas:
    convergence = np.empty([0, 1])
    for j in range(50):
        convergence = np.append(convergence,(perceptron(i, 20, 100)))
    convergence_av_basic = np.append(convergence_av_basic,np.mean(convergence))

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 24})
plt.plot(alphas, convergence_av_basic, marker='o', linestyle='-', color='b')
plt.xlabel('alpha')
plt.show()
