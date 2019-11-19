import warnings
import tensorflow as tf
from util.perturbation import *
import os

# reference:
# https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#training_loop
# only supports tensorflow 1.x

# forcing tensorflow to use cpu (if there is not enough graphics memory)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.enable_eager_execution()

warnings.filterwarnings(action='ignore', category=FutureWarning)

# train configurations
batchSize = 256
numEpochs = 3
globalStep = tf.Variable(0)
learningRate = tf.compat.v1.train.exponential_decay(learning_rate=5.0e-3, global_step=globalStep, decay_steps=80, decay_rate=0.95)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)

# raw data
print('loading MNIST data...')
train_X, train_Y = preprocess_mnist_data(*load_mnist_train_XY())
test_X, test_Y = preprocess_mnist_data(*load_mnist_test_XY())

test_X = test_X[:1000, :, :, :]
test_Y = test_Y[:1000]
test_X = tf.convert_to_tensor(test_X)
test_Y = tf.convert_to_tensor(test_Y)
test_YLabel = tf.argmax(test_Y, axis = 1, output_type=tf.int32)

# add perturbation
print('applying perturbation...')
pertMan = PerturbationMan(train_X, train_Y, get_perts())
#pertMan.show_content()
pertMan.create_tensor()

# the configuration of a new network
def get_new_network():
    network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=pertMan.get_perturbated_data(0).shape[1:], activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu', name="FC"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10) # no activation for the last layer
    ])

    return network

model = get_new_network()
# load std model for training
#model = tf.keras.models.load_model('nn_model/pretrained_model.dat')
#model.load_weights('nn_model/pretrained_model_weights.dat')

# the loss function of parallel networks
def loss_func(model, x, y):
    y_ = model(x)
    return tf.losses.softmax_cross_entropy(onehot_labels= y, logits=y_, reduction='none')


for epoch in range(numEpochs):

    modelAcc = tf.contrib.eager.metrics.Accuracy()

    totalBatches = int(np.ceil(train_X.shape[0] / batchSize))

    for batchInd in range(totalBatches):
        left = batchInd * batchSize
        right = min(train_X.shape[0], left + batchSize)

        losses = []

        batchX = batchY = None

        for dataInd in range(pertMan.get_num_groups()):
            # the last batchX and batchY are ground truth
            batchX, batchY = pertMan.get_tensor_slice(dataInd, left, right)
            loss = loss_func(model, batchX, batchY)
            losses.append(loss)

        # stack the loss in rows
        losses = tf.stack(losses, axis = 1)
        # find out the column with the greatest loss
        greatestLossPos = tf.argmax(losses, axis=1).numpy()

        thisBatchSize = batchX.shape[0]

        newBatchX = []

        # reorganize the batch, with the ones with highest loss
        for i in range(thisBatchSize):
            greatestLossInd = greatestLossPos[i]
            newBatchX.append(pertMan.get_tensor_x(greatestLossInd, left + i))

        newBatchX = tf.stack(newBatchX, axis=0)

        # compute loss and gradient
        with tf.GradientTape() as tape:
            lossVal = loss_func(model, newBatchX, batchY)
        grad = tape.gradient(lossVal, model.trainable_variables)

        optimizer.apply_gradients(zip(grad, model.trainable_variables), globalStep)

        batchPred = tf.argmax(model(test_X), axis=1, output_type=tf.int32)

        # compute accuracy
        modelAcc(batchPred, test_YLabel)

        print(f'batch {batchInd}/{totalBatches} (epoch: {epoch + 1} / {numEpochs}) acc: {modelAcc.result()}')



model.save('../nn_model/robust_model.dat')
#tf.contrib.saved_model.save_keras_model(model, 'robust_model.dat')
#model.save_weights('nn_model/robust_model_weights.dat')