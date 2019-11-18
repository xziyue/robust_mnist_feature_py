import matplotlib.pyplot as plt
import numpy as np
import warnings
from load_mnist import load_mnist_train_XY, load_mnist_test_XY, preprocess_mnist_data
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action='ignore', category=FutureWarning)
import pickle
import tensorflow as tf
from result_loader import load_robust_output, load_nonrobust_output

_, y = preprocess_mnist_data(*load_mnist_train_XY())

#data = load_robust_output()
#data = load_nonrobust_output()
with open('robust_dim_rec.bin', 'rb') as infile:
    data = pickle.load(infile)

train_X, test_X, train_Y, test_Y = train_test_split(data, y, test_size=0.1)

def get_new_network():
    network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=train_X.shape[1:], activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu', name="FC"),
        tf.keras.layers.Dropout(0.4),
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
          epochs=8,
          validation_data=(test_X, test_Y))

model.save('./nn_model/retrain_robust_dimrec.dat')