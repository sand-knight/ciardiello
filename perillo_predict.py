import numpy 
import tensorflow._api.v2.compat.v1 as tf
import utils

# tensorflow settings
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

# disable warnings
import warnings
warnings.filterwarnings("ignore")
import sys

data = numpy.genfromtxt('../sleeprawlive.csv')
#data = data[:20400]
data = data[:len(data)-len(data)%400]
#data = numpy.expand_dims(data, axis=0)
data = data.reshape(int(len(data)/400), 400)
data = numpy.expand_dims(data, axis=2)

print(data.shape)

def get_predictions(weights_format="weights-{:02d}.h5",
                    batch_size=32):
    modelpath = "components/deeper_fcn6"
    model = utils.get_model_from_json(modelpath)


    #model.load_weights(os.path.join(modelpath, weights_format.format(0)))
    model.load_weights("components/deeper_fcn6/weights")
    y_pred = model.predict(data)
    tf.keras.backend.clear_session()
    return y_pred[:, 0]

yy = get_predictions()
print(list(yy))

# print(simple_brueser(data))
# print(simple_mean_choi(data))
# print(diff_mean_choi(data))
# print(simple_mean_choe(data))
# print(diff_mean_choe(data))