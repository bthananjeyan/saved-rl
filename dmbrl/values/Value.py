from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import tensorflow as tf
import numpy as np

from dmbrl.misc.DotmapUtils import get_required_argument


class ValueFunction:
    def __init__(self, *args, **kwargs):
        """Creates class instance.
        """
        pass

    def train(self, obs_trajs, acs_trajs, costsum_trajs):
        """Trains this controller using lists of trajectories.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Resets this controller.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def value(self, obs):
        """Get value of this obs.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_logs(self, primary_logdir, iter_logdir):
        """Dumps logs into primary log directory and per-train iteration log directory.
        """
        raise NotImplementedError("Must be implemented in subclass.")

class DeepValueFunction(ValueFunction):
    def __init__(self, name, params):
        super().__init__(params)
        if params.gym_robotics:
            self.dO = params.env.observation_space.spaces['observation'].low.size
        else:
            self.dO = params.env.observation_space.shape[0]
        self.ac_ub, self.ac_lb = params.env.action_space.high, params.env.action_space.low
        self.ac_ub = np.minimum(self.ac_ub, params.get("ac_ub", self.ac_ub))
        self.ac_lb = np.maximum(self.ac_lb, params.get("ac_lb", self.ac_lb))
        self.update_fns = params.get("update_fns", [])
        self.per = params.get("per", 1)
        self.model = get_required_argument(
            params.model_init_cfg_val, "model_constructor", "Must provide a model constructor."
        )(name, params.model_init_cfg_val)
        self.model_train_cfg = params.get("model_train_cfg", {})
        self.buffer_limit = params.get("val_buffer_size", None)
        self.ign_var = params.get("ign_var", False)

        self.obs_preproc = params.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = params.get("obs_postproc", lambda obs, model_out: model_out)
        self.targ_proc = params.get("targ_proc", lambda val: val)

        self.save_all_models = params.log_cfg.get("save_all_models", False)
        opt_cfg = params.opt_cfg.get("cfg", {})

        # Controller state variables
        self.has_been_trained = params.prop_cfg.get("model_pretrained", False)
        self.train_in = np.array([]).reshape(0, self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        self.train_targs = np.array([])

        self.buffer = {
            "obs" : [],
            "cost": [],
            "next_obs": [],
            "val": [],
            "terminal": []
        }

        if self.model.is_tf_model:
            self.sy_cur_obs = tf.Variable(np.zeros(self.dO), dtype=tf.float32)
            # self.ac_seq = tf.placeholder(shape=[1, self.plan_hor*self.dU], dtype=tf.float32)
            # self.pred_cost, self.pred_traj = self._compile_cost(self.ac_seq, get_pred_trajs=True)
            # self.optimizer.setup(self._compile_cost, True)
            self.model.sess.run(tf.variables_initializer([self.sy_cur_obs]))
        else:
            raise NotImplementedError()

        if self.save_all_models:
            print("Value function will save all models. (Note: This may be memory-intensive.")
        else:
            print("Value function won't save all models")

    def copy_model_parameters(self):
        """
        Copies the model parameters of one estimator to another.
        Args:
          sess: Tensorflow session instance
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        estimator1 = self.model
        estimator2 = self.target
        e1_params = [t for t in self.model.all_vars]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in self.target.model.all_vars]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        self.model.sess.run(update_ops)

        checks = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            checks.append(tf.reduce_any(tf.logical_not(tf.equal(e1_v, e2_v))))
        res = self.model.sess.run(checks)
        assert np.sum(res) == 0
        # import IPython; IPython.embed()

    def train(self, obs_trajs, cost_trajs, next_obs_trajs, val_trajs, use_TD=False, gamma=1., terminal_states=[], copy_target=True):
        """Trains the value function : V: s --> value

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            val_trajs: A list of values.

        Returns: None.
        """
        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        self.buffer["obs"].extend(obs_trajs)
        self.buffer["cost"].extend(cost_trajs)
        self.buffer["next_obs"].extend(next_obs_trajs)
        self.buffer["val"].extend(val_trajs)

        self.buffer["terminal"].extend(terminal_states)

        buffer_limit = 200000
        for k in self.buffer.keys():
            self.buffer[k] = self.buffer[k][-buffer_limit:]

        if use_TD:
            self.train_in = np.array([]).reshape(0, self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
            self.train_targs = np.array([])
            for obs, cost, next_obs, val in zip(self.buffer["obs"], self.buffer["cost"], self.buffer["next_obs"], self.buffer["val"]):
                next_vals = gamma * self.target.value(self.obs_preproc(np.array(next_obs)))[0].ravel()
                new_train_in.append(self.obs_preproc(obs))
                new_train_targs.append(cost + next_vals) # no need to process cost, TODO: implement discounting
            for obs in self.buffer["terminal"]:
                new_train_in.append(np.expand_dims(self.obs_preproc(obs), axis=0))
                new_train_targs.append([0])
        else:
            for obs, val in zip(obs_trajs, val_trajs):
                new_train_in.append(self.obs_preproc(obs))
                new_train_targs.append(val) # no need to process val
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

        # Train the model
        if not self.has_been_trained:
            self.model.train(self.train_in, self.train_targs.reshape( (-1, 1) ), epochs=30)
        else:
            self.model.train(self.train_in, self.train_targs.reshape( (-1, 1) ), epochs=15)
        self.has_been_trained = True
        if copy_target:
            self.copy_model_parameters()

    def reset(self):
        if self.model.is_tf_model:
            for update_fn in self.update_fns:
                update_fn(self.model.sess)

    def value(self, obs_trajs, factored=False):
        if isinstance(obs_trajs, tf.Tensor):
            return self.model.create_prediction_tensors(obs_trajs, factored=factored)
        else:
            return self.model.predict(obs_trajs, factored=factored)

    def dump_logs(self, primary_logdir, iter_logdir):
        self.model.save(iter_logdir if self.save_all_models else primary_logdir)
