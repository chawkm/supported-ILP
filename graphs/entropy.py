import matplotlib.pyplot as plt

noise = [0, 5, 10, 15, 25, 40, 50, 60, 75, 85, 90, 100]

entropy =  [2.65, 3.571, 3.86, 4.316, 5.20, 6.79, 4.19, 5.66, 5.88, 5.78, 3.95, 2.95]

dilp =  [2.65, 3.571, 3.86, 4.316, 5.20, 6.79, 4.19, 5.66, 5.88, 5.78, 3.95, 2.95]

plt.plot(noise, entropy)
plt.ylabel('Entropy')
plt.xlabel('Noise')
# plt.title('Entropy of Hypothesis')
plt.show()