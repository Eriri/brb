import tensorflow as tf
import numpy as np
from util import read_oil, BaseBRB, RIMER
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import time
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def experiment(x, y, e):
    strategy = tf.distribute.MirroredStrategy()
    eva = np.load('eval%d.npy' % e)
    brb = []
    for base in range(10):
        with strategy.scope():
            b = load_model('base_%d_%d' % (e, base))
        brb.append(b)
    return brb


def main():
    x, y = read_oil()
    strategy = tf.distribute.MirroredStrategy()
    b = tf.keras.models.load_model('base_0_0')


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    main()
