import matplotlib.pyplot as plt
import numpy as np
import warnings
from load_mnist import load_mnist_train_XY, load_mnist_test_XY, preprocess_mnist_data
warnings.filterwarnings(action='ignore', category=FutureWarning)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K

assert K.image_data_format() == 'channels_last'

train_X, train_Y = preprocess_mnist_data(*load_mnist_train_XY())
test_X, test_Y = preprocess_mnist_data(*load_mnist_test_XY())


print(train_X.shape)
print(train_Y.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=train_X.shape[1:], activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax')) # the last layer has 10 perceptrons because there are 10 classes

model.summary()

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.sgd(lr=4.0e-4),
              metrics=['accuracy'])

model.fit(train_X, train_Y,
          batch_size=32,
          verbose=1,
          epochs=2,
          validation_data=(test_X, test_Y))

model.save('./nn_model/pretrained_model.dat')
model.save_weights('./nn_model/pretrained_model_weights.dat')