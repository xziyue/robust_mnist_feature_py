import numpy as np
import os
from util.result_loader import load_nonrobust_output


data = load_nonrobust_output()

saveDir = 'nonrobust'

# split the data into 10 segments
seg = 10
samplePerSeg = data.shape[0] // seg

for i in range(seg):
    left = i * samplePerSeg
    right = min((i + 1) * samplePerSeg, data.shape[0])
    np.save(os.path.join(saveDir, f'x_{i}'), data[left:right, :, :, :].squeeze())
    


