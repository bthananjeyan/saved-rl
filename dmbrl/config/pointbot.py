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

from dmbrl.env.pointbot_const import *


class PointBotConfigModule:
    ENV_NAME = "PointBot-v0"
    TASK_HORIZON = 100
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 15
    MODEL_IN, MODEL_OUT = 6, 4
    VALUE_IN, VALUE_OUT = 4, 1
    
    ALPHA_THRESH = 3
    HAS_CONSTRAINTS = True
    BETA_THRESH = 1
    
    NUM_DEMOS = 100
    DEMO_LOW_COST = None
    DEMO_HIGH_COST = None
    DEMO_LOAD_PATH = None
    LOAD_SAMPLES = False
    GYM_ROBOTICS = False
    SS_BUFFER_SIZE = 20000
    
    VAL_BUFFER_SIZE = 200000
    USE_VALUE = True

    def __init__(self, mode):
        self.ENV = gym.make(self.ENV_NAME)
        self.ENV.set_mode(mode)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        return obs

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        if isinstance(obs, np.ndarray):
            target = np.tile(GOAL_STATE, (len(obs), 1))
            if not HARD_MODE:
                return np.linalg.norm(np.subtract(target, obs), axis=1)
            return (np.linalg.norm(np.subtract(target, obs), axis=1) > GOAL_THRESH).astype(np.float32)
        else:
            target = np.tile(GOAL_STATE, (obs.shape[0], 1))
            target = tf.convert_to_tensor(target, dtype=tf.float32)
            if not HARD_MODE:
                return tf.norm(tf.subtract(target, obs), axis=1)
            return tf.cast(tf.norm(tf.subtract(target, obs), axis=1) > GOAL_THRESH, tf.float32)

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
        if not model_init_cfg.get("load_model", False):
            model.add(FC(500, input_dim=self.MODEL_IN, activation='swish', weight_decay=0.0001))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0005))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
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

CONFIG_MODULE = PointBotConfigModule
