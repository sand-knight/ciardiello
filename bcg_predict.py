import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import tensorflow._api.v2.compat.v1 as tf


import algorithms.heartrate as hr
import utils

# tensorflow settings
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

# disable warnings
import warnings
warnings.filterwarnings("ignore")

# x_data_train, y_data_train, groups_train = ...
# x_data_test, y_data_test, groups_test = ...

# dummy:
import sys

data = np.genfromtxt('data/experiments/sleeprawlive.csv', delimiter=';')
data = data.reshape( int(len(data)/400), 400)
x_data_test = np.expand_dims(data, axis=2)
x_data_test = x_data_test[200:458]



modelnames = [
    "models.deeper_fcn-x1",
    #"models.stacked_cnn_rnn_improved-x1",
]


def get_modelpath(modelname):
    return os.path.join("output", modelname)  # modify this if necessary


def get_predictions(modelname, weights_format="weights-{:02d}.h5",
                    batch_size=32):
    modelpath = get_modelpath(modelname)
    model = utils.get_model_from_json(modelpath)


    model.load_weights(os.path.join(modelpath,
                                    weights_format.format(0)))
    y_pred = model.predict(x_data_test)
    tf.keras.backend.clear_session()
    return y_pred[:, 0]

yy = get_predictions(modelnames[0])
print(yy)