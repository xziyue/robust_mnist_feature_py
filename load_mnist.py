import numpy as np
from bitstring import BitArray
import os

mnistImageSize = (28, 28)

def decode_idx_file(filename):

    with open(filename, 'rb') as infile:
        idxBuffer = infile.read()

    mgNum = [BitArray(bytes = idxBuffer[i : i + 1], length=8).int for i in range(4)]

    dtypeDict = {
        0x08 : np.uint8,
        0x09 : np.int8,
        0x0b: np.int16,
        0x0c: np.int32,
        0x0d: np.float32,
        0x0e: np.float64
    }

    dtype = dtypeDict[mgNum[2]]
    itemSize = dtype.itemsize

    numDim = mgNum[3]
    assert numDim > 0

    dims = []

    # read the dimensions
    for i in range(numDim):
        startByte = 4 + i * 4
        endByte = startByte + 4
        thisDim = BitArray(bytes= idxBuffer[startByte : endByte], length=32).int
        dims.append(thisDim)

    # read the entire buffer
    startByte = 4 + (numDim) * 4
    buffer = idxBuffer[startByte :]
    result = np.frombuffer(buffer, dtype)

    return result.reshape(dims)


mnistTrainFiles = [
    'train-images.idx3-ubyte',
    'train-labels.idx1-ubyte'
]

mnistTestFiles = [
    't10k-images.idx3-ubyte',
    't10k-labels.idx1-ubyte'
]

mnistDir = './dataset'

def load_mnist_train_XY():
    filename_X = os.path.join(mnistDir, mnistTrainFiles[0])
    filename_Y = os.path.join(mnistDir, mnistTrainFiles[1])
    X, Y = decode_idx_file(filename_X), decode_idx_file(filename_Y)
    assert X.shape[0] == Y.shape[0]
    return X, Y


def load_mnist_test_XY():
    filename_X = os.path.join(mnistDir, mnistTestFiles[0])
    filename_Y = os.path.join(mnistDir, mnistTestFiles[1])
    X, Y = decode_idx_file(filename_X), decode_idx_file(filename_Y)
    assert X.shape[0] == Y.shape[0]
    return X, Y

def to_categorical(Y):
    assert np.issubdtype(Y.dtype, np.integer)
    assert len(Y.shape) == 1
    minCate = Y.min()
    maxCate = Y.max()
    assert maxCate > minCate

    numDim = maxCate - minCate + 1

    result = np.zeros([Y.shape[0], numDim], np.int)

    for i in range(minCate, maxCate + 1):
        offset = i - minCate
        subVec = np.zeros(numDim, np.int)
        subVec[offset] = 1

        targetIndex = np.where(Y == i)[0]
        result[(targetIndex, ...)] = subVec

    return result


def preprocess_mnist_data(X, Y):
    X = X.astype(np.float32) /  255.0
    Y = to_categorical(Y)
    X = X.reshape(list(X.shape) + [1])

    return X ,Y