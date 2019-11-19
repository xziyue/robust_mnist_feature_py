import pickle
from util.load_mnist import load_mnist_train_XY
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import pairwise_distances

x, y = load_mnist_train_XY()

'''
x_r = load_robust_output()
x_nr = load_nonrobust_output()
'''


with open('../robust_morph.bin', 'rb') as infile:
    x_r = pickle.load(infile)

with open('../nonrobust_morph.bin', 'rb') as infile:
    x_nr = pickle.load(infile)

numSamples = x.shape[0]

x = x.reshape((numSamples, -1)).astype(np.float)/255.0
x_r = x_r.reshape((numSamples, -1))
x_nr = x_nr.reshape((numSamples, -1))


locations = [np.where(y == i)[0] for i in range(10)]

result = []

for i in range(10):
    sampleMat = x[locations[i], ...]
    sampleMat_r = x_r[locations[i], ...]
    sampleMat_nr = x_nr[locations[i], ...]

    toDelete = [j * j for j in range(sampleMat.shape[0])]

    dist = np.delete(pairwise_distances(sampleMat).flatten(), toDelete)
    mean_ = np.mean(dist)
    std_ = np.std(dist)
    norm_ = norm(mean_, std_)

    dist_r = np.delete(pairwise_distances(sampleMat_r).flatten(), toDelete)
    mean_r = np.mean(dist_r)
    std_r = np.std(dist_r)
    norm_r = norm(mean_r, std_r)

    dist_nr = np.delete(pairwise_distances(sampleMat_nr).flatten(), toDelete)
    mean_nr = np.mean(dist_nr)
    std_nr = np.std(dist_nr)
    norm_nr = norm(mean_nr, std_nr)

    # show histogram
    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
    ax[0].hist(dist_r, bins=100, density=True)
    ax[1].hist(dist_nr, bins=100, density=True)
    ax[2].hist(dist, bins=100, density=True)

    res = {
        'figure' : fig,
        'dist_x' : norm_,
        'dist_x_r' : norm_r,
        'dist_x_nr' : norm_nr
    }

    result.append(res)

    '''
    x1, x2 = ax[0].get_xlim()
    xVals = np.linspace(x1, x2, 200)
    ax[0].plot(xVals, norm_r.pdf(xVals))
    ax[1].plot(xVals, norm_nr.pdf(xVals))
    '''


with open('../pdist_result.bin', 'wb') as outfile:
    pickle.dump(result, outfile)







