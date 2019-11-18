from result_loader import load_robust_output, load_nonrobust_output
from load_mnist import load_mnist_train_XY
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from load_mnist import to_categorical
import tensorflow as tf



_, y = load_mnist_train_XY()
x_r = load_robust_output()
x_nr = load_nonrobust_output()

# find samples for each label
locations = [np.where(y == i)[0] for i in range(10)]

network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(2, (3, 3), input_shape=x_r.shape[1:], activation='relu'),
        tf.keras.layers.Conv2D(3, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(30, activation='relu', name="FC"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax') # no activation for the last layer
])

for i in range(1, 10):
    nrSamples = x_nr[locations[i], ...]
    rSamples = x_r[locations[i], ...]

    samples = np.concatenate([nrSamples, rSamples], axis=0)
    labels = np.asarray([0] * nrSamples.shape[0] + [1] * rSamples.shape[0])
    labels = to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=0x123efddd)

    network.summary()
    network.compile(loss = 'categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=3.0e-5),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

    network.fit(x_train, y_train,
                batch_size=32,
                verbose=1,
                epochs=20,
                validation_data=(x_test, y_test))

    network.save('nn_model/diff_{}.dat'.format(i))

    exit(0)


'''
# flatten features for SVM
x_r = x_r.reshape((x_r.shape[0], -1))
x_nr = x_r.reshape((x_nr.shape[0], -1))

# train SVM
for i in range(10):
    nrSamples = x_nr[locations[i], ...]
    rSamples = x_r[locations[i], ...]

    samples = np.concatenate([nrSamples, rSamples], axis=0)
    labels = [0] * nrSamples.shape[0] + [1] * rSamples.shape[0]

    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=0x123efddd)
    print(x_train.shape)
    print(x_test.shape)

    svm = SVC(verbose=1)
    svm.fit(x_train, y_train)
    print('train score for {}:{}'.format(i, svm.score(x_train, y_train)))
    print('test score for {}:{}'.format(i, svm.score(x_test, y_test)))
'''