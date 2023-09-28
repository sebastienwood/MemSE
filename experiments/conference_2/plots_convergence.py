import matplotlib.pyplot as plt
import numpy as np

acc = np.load('acc_convergence.npy')
power = np.load('power_convergence.npy')

plt.plot(acc.T)
plt.title('Accuracy convergence analysis')
plt.ylabel('Average accuracy')
plt.xlabel('Minibatch')
plt.savefig('acc_convergence.png')
plt.close()

plt.plot(power.T)
plt.title('Power convergence analysis')
plt.ylabel('Average power')
plt.xlabel('Minibatch')
plt.savefig('power_convergence.png')
plt.close()