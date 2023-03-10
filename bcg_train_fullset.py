import os

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import tensorflow._api.v2.compat.v1 as tf
import sklearn
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from models import deeper_fcn as architecture
import algorithms.heartrate as hr
import utils

# tensorflow settings
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

# to not exhaust memory (later jobs have been moved from 4 to 2)
from tensorflow.python.framework.config import set_memory_growth
tf.compat.v1.disable_v2_behavior()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)




# load data
# x_data_train, y_data_train, groups_train = ...
# dummy:
modelpath = "output"
train_size = 2938
n_groups = 28
data = np.genfromtxt('data/experiments/sleeprawlive.csv', delimiter=',')
# print the shape of the data
print("data len", data.shape)
data = data[:len(data)-len(data)%400]
# reshape the data to be 2D array with 400 columns and 5 rows   (400, 5)
data = data.reshape( int(len(data)/400), 400)
# print the shape of the data
print("data reshaped", data.shape)
x_data_train = np.expand_dims(data, axis=2)
#y_train section
data = np.genfromtxt('data/experiments/gold.csv', delimiter=',')
print("gold data len", len(data))
# compute average each 8 elements in the array
data = data[:len(data)-len(data)%16]

y_data_train = np.mean(data.reshape(int(len(data)/16), 16), axis=1)
print("gold data reshaped", y_data_train.shape)
# print first 2 rows of the data
print(y_data_train[:2])
y_data_train= y_data_train[:2938]
groups_train = np.sort(np.random.randint(n_groups, size=train_size))

print(x_data_train.shape, y_data_train.shape, groups_train.shape)

#y_data_train = y_data_train.values
#x_data_train= x_data_train.values




enlarge = 1
model_params = dict(metrics=["mae", "mape"], enlarge=enlarge)
fit_params = dict(epochs=30, verbose=2)  # set epochs between 30 and 75

modelname = (architecture.__name__ + "-x{}".format(enlarge))
modelpath = os.path.join("output", modelname)


# train one model on entire training set

model = architecture.create(**model_params)
r = model.fit(x_data_train, y_data_train, **fit_params)
model.save_weights(os.path.join(modelpath, "final", "weights-00.h5"))
tf.keras.backend.clear_session()