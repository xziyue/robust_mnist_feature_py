from util.perturbation import *
from sklearn.metrics import accuracy_score
import tensorflow as tf

model = tf.keras.models.load_model('../nn_model/robust_model.dat')

pertMan = get_pertubated_test_data()

for i in range(pertMan.get_num_groups()):
    pred_Y = np.argmax(model.predict(pertMan.data[i]), axis=1)
    Y = np.argmax(pertMan.Y, axis=1)
    print('acc: {}'.format(accuracy_score(Y, pred_Y)))
