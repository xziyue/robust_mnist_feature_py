from util.result_loader import load_nonrobust_output
import pickle
import numpy as np

'''
# PCA noise cancellation
x_r = load_robust_output()
x_r = x_r.reshape((x_r.shape[0], -1))

scaler = StandardScaler()
x_r_tf = scaler.fit_transform(x_r)

pca = PCA(n_components=400)
intermediate = pca.fit_transform(x_r_tf)
x_denoised = pca.inverse_transform(intermediate)
x_res = scaler.inverse_transform(x_denoised).reshape((x_r.shape[0], 28, 28, 1))
x_res = np.clip(x_res, 0.0, 1.0)


with open('../robust_dim_rec.bin', 'wb') as outfile:
    pickle.dump(x_res, outfile)
'''

'''
# morphology on robust
from skimage.morphology import opening, disk, square
x_r = load_robust_output().squeeze()
# convert to mask
x_r_morph = []
for i in range(x_r.shape[0]):
    img = x_r[i, :, :].copy()
    mask = img > 0.3
    mask = opening(mask, square(2))
    invMask = np.bitwise_not(mask)
    img[invMask] = 0.0
    x_r_morph.append(img)
x_r_morph = np.stack(x_r_morph, axis=0)
x_r_morph = x_r_morph.reshape(list(x_r.shape) + [1])

with open('../robust_morph.bin', 'wb') as outfile:
    pickle.dump(x_r_morph, outfile)
'''

# morphology on nonrobust
from skimage.morphology import opening, square
x_nr = load_nonrobust_output().squeeze()
# convert to mask
x_nr_morph = []
for i in range(x_nr.shape[0]):
    img = x_nr[i, :, :].copy()
    mask = img > 0.3
    mask = opening(mask, square(2))
    invMask = np.bitwise_not(mask)
    img[invMask] = 0.0
    x_nr_morph.append(img)
x_r_morph = np.stack(x_nr_morph, axis=0)
x_r_morph = x_r_morph.reshape(list(x_nr.shape) + [1])

with open('../nonrobust_morph.bin', 'wb') as outfile:
    pickle.dump(x_r_morph, outfile)