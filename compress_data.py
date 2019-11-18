import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from load_mnist import load_mnist_train_XY
from result_loader import load_robust_output, load_nonrobust_output


data = load_nonrobust_output()

saveDir = 'nonrobust'

# split the data into 10 segments
seg = 10
samplePerSeg = data.shape[0] // seg

for i in range(seg):
    left = i * samplePerSeg
    right = min((i + 1) * samplePerSeg, data.shape[0])
    np.save(os.path.join(saveDir, f'x_{i}'), data[left:right, :, :, :].squeeze())
    


