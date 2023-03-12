import os
#from joblib import Parallel, delayed
import numpy, pandas
import tensorflow._api.v2.compat.v1 as tensorflow
#import sklearn
#from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from scipy import signal
from models import deeper_fcn

tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
tensorflow.keras.backend.set_session(tensorflow.Session(config=config))

from tensorflow.python.framework.config import set_memory_growth
tensorflow.compat.v1.disable_v2_behavior()
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

train_size = 2938
n_groups = 28

data = numpy.genfromtxt("../raw_2022-03-14_bcg.csv")
data = data[:len(data)-len(data)%400]
data = data.reshape( int(len(data)/400),400)
bcg = numpy.expand_dims(data, axis=2)
print("bcg", data.shape, bcg.shape)

train_size = bcg.shape[0]

data=numpy.genfromtxt("../gold.csv")
#gold = signal.resample(data, int(len(data)/32))
data = data[:len(data)-len(data)%32]
gold = numpy.mean(data.reshape(int(len(data)/16), 16), axis=1)
print("gold", data.shape, gold.shape)
gold = gold[:train_size]

groups_train = numpy.sort(numpy.random.randint(n_groups, size=train_size))
print(bcg.shape, gold.shape, groups_train.shape)

enlarge=1
model_params = dict(metrics=["mae", "mape"], enlarge=enlarge)
fit_params = dict(epochs=66, verbose=2)

model = deeper_fcn.create(**model_params)

with open("components/deeper_fcn6/model.json", 'w') as fp:
    fp.write(model.to_json())

model.fit(bcg, gold, **fit_params)
model.save_weights("components/deeper_fcn6/weights")
tensorflow.keras.backend.clear_session()



