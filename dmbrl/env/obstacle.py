import os
import pickle

import os.path as osp
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from gym import Env
from gym import utils
from gym.spaces import Box

class Obstacle:
    def __init__(self, boundsx, boundsy, penalty=100):
        self.boundsx = boundsx
        self.boundsy = boundsy
        self.penalty = 1


    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return (self.boundsx[0] <= x[0] <= self.boundsx[1] and self.boundsy[0] <= x[2] <= self.boundsy[1]) * self.penalty
        else:
            condition1 = tf.logical_and(
                self.boundsx[0] <= x[:, :, :, 0], x[:, :, :, 0] <= self.boundsx[1]
            )
            condition2 = tf.logical_and(
                self.boundsy[0] <= x[:, :, :, 2], x[:, :, :, 2] <= self.boundsy[1]
            )

            condition = tf.logical_and(condition1, condition2)

            x_sum = tf.reduce_sum(x, axis=-1) # This is a bad way to do this but works for now, just using this for shape
            return tf.where(condition, self.penalty * tf.ones_like(x_sum), tf.zeros_like(x_sum))
            # return (self.boundsx[0] <= x[0] <= self.boundsx[1] and self.boundsy[0] <= x[2] <= self.boundsy[1]) * 10000

class ComplexObstacle(Obstacle):

    def __init__(self, bounds):
        self.obs = []
        for boundsx, boundsy in bounds:
            self.obs.append(Obstacle(boundsx, boundsy))

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return np.max([o(x) for o in self.obs])
        else:
            return sum([o(x) for o in self.obs])