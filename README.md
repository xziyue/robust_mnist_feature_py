# Analyzing Robust Features From MNIST

website: <https://github.com/xziyue/robust_mnist_feature_py>

## TODOs:

- [x] Implement robust training

- [x] Implement a sufficient amount of perturbation

- [x] Compare performance of std model and robust model

- [x] Implement gradient descent for reconstructing features

## Current Problems

- [x] Convergence: the robust model does not seem to converge well (may need to pretrain the model first)

- [ ] Why does horizontal lines hurt accuracy more significantly than vertical lines?

## Goals

- [ ] Is it possible to synthesize "robust" features directly?
- [ ] Is it possible to differentiate nonrobust and robust features blindly?
- [ ] Is is possible to create perturbation that leads to human-readable robust features?

## The dataset

The MNIST datset is available at <http://yann.lecun.com/exdb/mnist/>.

If you would like to run this script on your computer, go to `/dataset` folder and uncompress all the dataset files to that folder.

## Test results

The perturbated image samples can be seen in figure below. The last column is ground truth. The group IDs correspond to the order
of images in the figure.

![perturbated image samples](images/data_sample.png)


| Group Id | Std Accuracy | Robust Accuracy|
|:---:|:---:|:---:|
|1| 0.829 | 0.968 |
|2| 0.549 | 0.967 |
|3| 0.808 | 0.969 |
|4| 0.727 | 0.950 |
|5| 0.977 | 0.972 |

Running standard training over reconstructed datasets:

| Group Id | Robust Accuracy | Nonrobust Accuracy|
|:---:|:---:|:---:|
|1| 0.792 | 0.856 |
|2| 0.822 | 0.434 |
|3| 0.908 | 0.865 |
|4| 0.876 | 0.657 |
|5| 0.960 | 0.954 |

## Reconstruction

The reconstructed features can be downloaded from this [repo](https://github.com/xziyue/MNIST_Features).

|Original|Reconstruction (Robust)|Reconstruction (Nonrobust)|
|:---:|:---:|:---:|
|![](images/original_0.png)|![](images/robust_recon_0.png)|![](images/nonrobust_recon_0.png)|
|![](images/original_1.png)|![](images/robust_recon_1.png)|![](images/nonrobust_recon_1.png)|
|![](images/original_2.png)|![](images/robust_recon_2.png)|![](images/nonrobust_recon_2.png)|
|![](images/original_3.png)|![](images/robust_recon_3.png)|![](images/nonrobust_recon_3.png)|
|![](images/original_4.png)|![](images/robust_recon_4.png)|![](images/nonrobust_recon_4.png)|
|![](images/original_5.png)|![](images/robust_recon_5.png)|![](images/nonrobust_recon_5.png)|
|![](images/original_6.png)|![](images/robust_recon_6.png)|![](images/nonrobust_recon_6.png)|
|![](images/original_7.png)|![](images/robust_recon_7.png)|![](images/nonrobust_recon_7.png)|
|![](images/original_8.png)|![](images/robust_recon_8.png)|![](images/nonrobust_recon_8.png)|
|![](images/original_9.png)|![](images/robust_recon_9.png)|![](images/nonrobust_recon_9.png)|

## Noise Cancellation on Features

|Denoised Robust Features|Denoised Nonrobust Features|
|:---:|:---:|
|![](images/robust_recon_morph_0.png)|![](images/nonrobust_recon_morph_0.png)|
|![](images/robust_recon_morph_1.png)|![](images/nonrobust_recon_morph_1.png)|
|![](images/robust_recon_morph_2.png)|![](images/nonrobust_recon_morph_2.png)|
|![](images/robust_recon_morph_3.png)|![](images/nonrobust_recon_morph_3.png)|
|![](images/robust_recon_morph_4.png)|![](images/nonrobust_recon_morph_4.png)|
|![](images/robust_recon_morph_5.png)|![](images/nonrobust_recon_morph_5.png)|
|![](images/robust_recon_morph_6.png)|![](images/nonrobust_recon_morph_6.png)|
|![](images/robust_recon_morph_7.png)|![](images/nonrobust_recon_morph_7.png)|
|![](images/robust_recon_morph_8.png)|![](images/nonrobust_recon_morph_8.png)|
|![](images/robust_recon_morph_9.png)|![](images/nonrobust_recon_morph_9.png)|

## File description:

Remember to add the root dir to PYTHONPATH.

*I am doing a bunch of crazy experiments right now, there are many undocumented files in the repo.*

- `util` folder:
    - `perturbation.py`: creates and manages perturbations
    - `load_mnist.py`: loading data from MNIST idx format (need to correct endianess if the data format has sizes greater than 1 byte)
- `train` folder: neural network training scripts
    - `train_std_model.py`: trains standard model
    - `train_pretrained_model`: trains a pretrain model as initial weights for robust model
    - `train_robust_model.py`: trains the robust model
- `test` folder: test the performance of models
    - `test_std_model`: tests the performance of std model on adversarial dataset
    - `test_robust_model`: tests the performance of robust model on adversarial dataset
- `reconstruct` folder: reconstructing the features from models
- `misc` folder: some ongoing experiments


## References

- Ilyas, Andrew, et al. "Adversarial examples are not bugs, they are features." *arXiv preprint arXiv:1905.02175* (2019).

