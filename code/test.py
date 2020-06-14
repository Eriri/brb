import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from util import *
import os


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.ax = tf.Variable(np.random.uniform(size=(5, 2,)))
        self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MSE, metrics=['mae'])


os.system('paplay beep.wav')
