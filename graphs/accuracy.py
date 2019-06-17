import matplotlib.pyplot as plt

noise = [0, 5, 10, 15, 25, 40, 50, 60, 75, 85, 90, 100]

entropy =  [0, 0, 0, 0.375, 0.375, 0.75, 0.6875, 1, 1, 1, 1, 1]

dilp =  [0, 0, 0, 0.002, 0.029, 0.127, 0.127, 0.174, 0.199, 0.202, 0.203, 0.182]

noise_ilasp = [0, 5, 10, 15, 25, 40, 50, 65, 75, 85, 90, 100]
ilasp =  [0, 0, 0.05, 0.0, 0.2, 0.18, 0.4, 0.6, 0.85, 0.9, 0.95, 1.0]


plt.plot(noise, entropy, label='dSILP')
plt.plot(noise, dilp, label='dILP')
plt.plot(noise_ilasp, ilasp, label='ILASP3')

plt.ylabel('MSE')
plt.xlabel('Noise')
plt.legend()
# plt.title('Entropy of Hypothesis')
plt.show()
