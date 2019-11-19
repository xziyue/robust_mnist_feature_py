import numpy as np
import pickle

def load_robust_output():
    result = []
    for i in range(6):
        fn = '../robust_out_{}.bin'.format(i)
        with open(fn, 'rb') as infile:
            data = pickle.load(infile)
            result.extend(data)
    return np.concatenate(result, axis=0)


def load_nonrobust_output():
    result = []
    for i in range(5):
        fn = '../nonrobust_out_{}.bin'.format(i)
        with open(fn, 'rb') as infile:
            data = pickle.load(infile)
            result.extend(data)
    return np.concatenate(result, axis=0)
