from perturbation import *
import tensorflow as tf
from result_loader import load_nonrobust_output, load_robust_output

# forcing tensorflow to use cpu (if there is not enough graphics memory)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.enable_eager_execution()

# raw data
_, y = load_mnist_train_XY()
locations = [np.where(y == i)[0] for i in range(10)]

# load reconstructed features
feat_r = load_robust_output()
feat_nr = load_nonrobust_output()

def get_num_samples(digit):
    return locations[digit].size

maxEpoches = 100
learningRate = 1.0
decay = 0.996

def reconstruct_diff(digit, classId, sampleId):
    assert 0 <= sampleId < get_num_samples(digit)

    # load the model
    model = tf.keras.models.load_model('nn_model/diff_{}.dat'.format(digit))
    intermediateModel = tf.keras.Model(inputs=model.input, outputs=model.get_layer('FC').output)

    sampleLoc = locations[digit][sampleId]
    assert y[sampleLoc] == digit
    if classId == 0:
        x = feat_nr[sampleLoc: sampleLoc + 1, :, :, :]
    elif classId == 1:
        x = feat_r[sampleLoc: sampleLoc + 1, :, :, :]
    else:
        raise AttributeError('unknown class id')

    tensor_x = tf.convert_to_tensor(x)
    modelPredictReal = intermediateModel.predict(tensor_x)

    '''
    # initialize with a random image in the dataset
    while True:
        sourceClassId = np.random.randint(0, 1)
        selectId = np.random.randint(0, get_num_samples(digit) - 1)
        if sourceClassId != classId:
            break
        if selectId != sampleLoc:
            break

    if sourceClassId == 0:
        start_x = feat_nr[selectId: selectId + 1, :, :, :]
    else:
        start_x = feat_r[selectId: selectId + 1, :, :, :]
    '''

    start_x = np.clip(np.random.normal(0.5, 0.1, x.size), 0, 1).astype(np.float32).reshape(x.shape)
    start_x = tf.convert_to_tensor(start_x)

    lastLoss = 0.0
    lossDiff = 1.0e6

    for epoch in range(maxEpoches):
        if abs(lossDiff) < 0.001:
            break

        with tf.GradientTape() as tape:
            tape.watch(start_x)
            modelPredictNow = intermediateModel(start_x)
            loss = tf.norm(modelPredictNow - modelPredictReal)
        gradient = tape.gradient(loss, start_x)
        # normalize the gradient
        gradientNorm = tf.norm(gradient)
        gradient = tf.math.divide(gradient, gradientNorm)

        # apply the gradient
        start_x = tf.math.subtract(start_x, tf.math.multiply(gradient, learningRate * (decay ** epoch)))
        # clip to 0, 1
        start_x = tf.clip_by_value(start_x, 0.0, 1.0)

        lossVal = loss.numpy()
        # update loss value
        lossDiff = lossVal - lastLoss
        lastLoss = lossVal

        #print(lossVal, lossDiff)

    return start_x.numpy()

for i in range(10):
    feat1 = reconstruct_diff(1, 0, i).squeeze()
    feat2 = reconstruct_diff(1, 1, i).squeeze()
    plt.subplot(121)
    plt.imshow(feat1, cmap='jet')
    plt.subplot(122)
    plt.imshow(feat2, cmap='jet')
    plt.colorbar()
    plt.show()
