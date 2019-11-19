import tensorflow as tf
from util.perturbation import *
from sklearn.metrics import accuracy_score

#model = tf.keras.models.load_model('../nn_model/retrain_robust_model.dat')
#model = tf.keras.models.load_model('../nn_model/retrain_nonrobust_model.dat')
#model = tf.keras.models.load_model('../nn_model/std_model_1.dat')

model = tf.keras.models.load_model('../nn_model/retrain_nonrobust_morph.dat')


pertMan = get_pertubated_test_data()
#pertMan.show_content()

for i in range(pertMan.get_num_groups()):
    pred_Y = np.argmax(model.predict(pertMan.data[i]), axis=1)
    Y = np.argmax(pertMan.Y, axis=1)
    print('acc: {}'.format(accuracy_score(Y, pred_Y)))
