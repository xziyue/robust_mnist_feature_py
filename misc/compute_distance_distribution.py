from util.load_mnist import load_mnist_train_XY
import numpy as np
import matplotlib.pyplot as plt

from util.result_loader import load_nonrobust_output, load_robust_output

x, y = load_mnist_train_XY()

x_r = load_robust_output()
x_nr = load_nonrobust_output()


'''
with open('robust_morph.bin', 'rb') as infile:
    x_r = pickle.load(infile)

with open('nonrobust_morph.bin', 'rb') as infile:
    x_nr = pickle.load(infile)
    
'''

numSamples = x.shape[0]

x = x.reshape((numSamples, -1))
x_r = x_r.reshape((numSamples, -1))
x_nr = x_nr.reshape((numSamples, -1))

diff_r = x - x_r
norm_r = np.linalg.norm(diff_r, axis=1)

diff_nr = x - x_nr
norm_nr = np.linalg.norm(diff_nr, axis=1)

fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
ax[0].hist(norm_r, bins=100)
ax[1].hist(norm_nr, bins=100)
plt.show()

'''
locations = [np.where(y == i)[0] for i in range(10)]

for i in range(10):
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)

    ax[0].hist(norm_r[locations[i]], bins=100)
    ax[1].hist(norm_nr[locations[i]], bins=100)

    plt.show()
'''
