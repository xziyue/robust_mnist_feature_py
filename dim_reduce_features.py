from result_loader import load_robust_output, load_nonrobust_output
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import numpy as np

x_r = load_robust_output()
x_r = x_r.reshape((x_r.shape[0], -1))

scaler = StandardScaler()
x_r_tf = scaler.fit_transform(x_r)

pca = PCA(n_components=400)
intermediate = pca.fit_transform(x_r_tf)
x_denoised = pca.inverse_transform(intermediate)
x_res = scaler.inverse_transform(x_denoised).reshape((x_r.shape[0], 28, 28, 1))
x_res = np.clip(x_res, 0.0, 1.0)

with open('robust_dim_rec.bin', 'wb') as outfile:
    pickle.dump(x_res, outfile)
