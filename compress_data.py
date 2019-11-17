import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from load_mnist import load_mnist_train_XY



with open('robust_features.bin', 'rb') as infile:
    data = pickle.load(infile)

data = np.concatenate(data, axis=0)

saveDir = 'robust'

# split the data into 10 segments
seg = 10
samplePerSeg = data.shape[0] // seg


for i in range(seg):
    left = i * samplePerSeg
    right = min((i + 1) * samplePerSeg, data.shape[0])
    np.save(os.path.join(saveDir, f'x_{i}'), data[left:right, :, :, :].squeeze())
    


