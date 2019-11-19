from util.perturbation import *
import tensorflow as tf
import warnings

# forcing tensorflow to use cpu (if there is not enough graphics memory)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.enable_eager_execution()

# raw data
#print('loading MNIST data...')
train_X, train_Y = preprocess_mnist_data(*load_mnist_train_XY())

warnings.filterwarnings(action='ignore', category=FutureWarning)

# load the model
model = tf.keras.models.load_model('../nn_model/std_model.dat')
intermediateModel = tf.keras.Model(inputs = model.input, outputs = model.get_layer('FC').output)


maxEpoches = 100
learningRate = 1.0
decay = 0.996

def reconstruct_feature(featureId):
    x = train_X[featureId : featureId + 1, :, :, :]
    tensor_x = tf.convert_to_tensor(x)
    modelPredictReal = intermediateModel.predict(tensor_x)

    # initialize with a random image in the dataset
    selectId = -1
    while True:
        selectId = np.random.randint(0, train_X.shape[0] - 1)
        if selectId != featureId:
            break
    start_x = train_X[selectId: selectId + 1, :, :, :]

    #start_x = np.clip(np.random.normal(0.5, 0.01, x.size), 0, 1).astype(x.dtype).reshape(x.shape)
    #start_x = train_X[200 : 200 + 1, :, :, :]
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
