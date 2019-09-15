import matplotlib.pyplot as plt
import numpy as np
import warnings
from load_mnist import load_mnist_train_XY, load_mnist_test_XY, preprocess_mnist_data
warnings.filterwarnings(action='ignore', category=FutureWarning)

import tensorflow as tf

train_X, train_Y = preprocess_mnist_data(*load_mnist_train_XY())
test_X, test_Y = preprocess_mnist_data(*load_mnist_test_XY())

def get_new_network():
    network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=train_X.shape[1:], activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu', name="FC"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax') # no activation for the last layer
    ])

    return network

print(train_X.shape)
print(train_Y.shape)

model = get_new_network()

model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=3.0e-5),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(train_X, train_Y,
          batch_size=32,
          verbose=1,
          epochs=6,
          validation_data=(test_X, test_Y))

model.save('./nn_model/std_model.dat')