# Extracting Robust Features From MNIST

## TODOs:
-[x] Implement robust training
-[x] Implement a sufficient amount of perturbation
-[ ] Implement gradient descent for reconstructing features

## Current Problems

-[ ] Convergence: the robust model does not seem to converge well (may need to pretrain the model first)

## The dataset

The MNIST datset is available at <http://yann.lecun.com/exdb/mnist/>.

If you would like to run this script on your computer, go to `/dataset` folder and uncompress all the dataset files to that folder.

## File description:
- `perturbation.py`: creates and manages perturbations
- `train_robust_model.py`: trains the robust model
- `load_mnist.py`: loading data from MNIST idx format (need to correct endianess if the data format has sizes greater than 1 byte)
- `test_mnist.py`: testing script