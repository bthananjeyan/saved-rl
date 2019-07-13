from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env

class PickAndPlaceConfigModule:
    ENV_NAME = "MBRL-PickAndPlace-v1"
    TASK_HORIZON = 50
    NTRAIN_ITERS = 200
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 10
    MODEL_IN, MODEL_OUT = 12, 8
    VALUE_IN, VALUE_OUT = 8, 1
    CLONE_IN, CLONE_OUT = 8, 4
    GP_NINDUCING_POINTS = 200

    ALPHA_THRESH = 0.05
    HAS_CONSTRAINTS = False
    BETA_THRESH = 1
    
    NUM_DEMOS = 100
    DEMO_LOW_COST = 0
    DEMO_HIGH_COST = 100
    DEMO_LOAD_PATH = "experts/fetchpickandplace/data_fetch_random_100.p"
    LOAD_SAMPLES = True
    GYM_ROBOTICS = True
    SS_BUFFER_SIZE = 5000
    
    VAL_BUFFER_SIZE = 100000
    USE_VALUE = True

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.ENV.reset()
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 10000
            },
            "CEM": {
                "popsize": 2000,
                "num_elites": 100,
                "max_iters": 7,
                "alpha": 0.1
            }
        }
        self.UPDATE_FNS = [self.update_goal]

        self.goal = tf.Variable(self.ENV.goal, dtype=tf.float32)
        self.SESS.run(self.goal.initializer)

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def update_goal(self, sess=None):
        if sess is not None:
            self.goal.load(self.ENV.goal, sess)

    def obs_cost_fn(self, obs):
        if isinstance(obs, np.ndarray):
            d = np.linalg.norm(obs[:, :3], axis=-1)
#            return (d > self.ENV.distance_threshold).astype(np.float32)

            # grip = obs[:,3:5].sum(axis=-1)
            d2 = np.linalg.norm(obs[:, 5:8], axis=-1)
            return (d2 > self.ENV.distance_threshold).astype(np.float32)
            # return (np.add(d, d2)).astype(np.float32)
        else:
            d = tf.norm(obs[:,:3], axis=-1)
#            return tf.cast(d > self.ENV.distance_threshold, tf.float32)
            d2 = tf.norm(obs[:,5:8], axis=-1)
            # grip = tf.reduce_sum(obs[:,3:5], axis=-1)
            return tf.cast(d2 > self.ENV.distance_threshold, tf.float32)
            # return tf.cast(tf.add(tf.cast(d, tf.float32), d2), tf.float32)
 
    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0.0 * np.sum(np.square(acs), axis=1)
        else:
            return 0.0 * tf.reduce_sum(tf.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        model.env_copy = self.env_copy()
        model.env_name = self.ENV_NAME
        model.env = self.env_copy()()
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.00025))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(self.MODEL_OUT, weight_decay=0.00075))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.00075})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model

    def value_nn_constructor(self, name, model_init_cfg_val):
        model = get_required_argument(model_init_cfg_val, "model_class", "Must provide model class")(DotMap(
            name=name, num_networks=get_required_argument(model_init_cfg_val, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg_val.get("load_model", False),
            model_dir=model_init_cfg_val.get("model_dir", None)
        ))
        if not model_init_cfg_val.get("load_model", False):
            model.add(FC(500, input_dim=self.VALUE_IN, activation='swish', weight_decay=0.0001))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(self.VALUE_OUT, weight_decay=0.0005))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001}, suffix = "val")
        return model

    def env_copy(self):
        return lambda: gym.make(self.ENV_NAME)


CONFIG_MODULE = PickAndPlaceConfigModule