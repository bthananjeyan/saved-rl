from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import tensorflow as tf
import numpy as np
from scipy.io import savemat

from .Controller import Controller
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.optimizers import RandomOptimizer, CEMOptimizer

class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, params):
        """Creates class instance.

        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                    .obstacle (function defined in environment): Essentially an indicator function
                        which determines whether any given state is constraint violating
                .update_fns (list<func>): A list of functions that will be invoked
                    (possibly with a tensorflow session) every time this controller is reset.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .has_constraints (bool): Indicates whether the environment has constraints
                .alpha_thresh (float): safety density model threshold
                .beta_thresh (float): constraint satisfaction probability threhsold
                .use_value (bool): Indicates whether to use value function
                .value_func (DeepValueFunction): Value function used for SAVED algorithm
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM, Random].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                .log_cfg
                    .save_all_models (bool): (optional) If True, saves models at every iteration.
                        Defaults to False (only most recent model is saved).
                        Warning: Can be very memory-intensive.
                    .log_traj_preds (bool): (optional) If True, saves the mean and variance of predicted
                        particle trajectories. Defaults to False.
                    .log_particles (bool) (optional) If True, saves all predicted particles trajectories.
                        Defaults to False. Note: Takes precedence over log_traj_preds.
                        Warning: Can be very memory-intensive
        """
        super().__init__(params)
        self.has_constraints = params.has_constraints
        if self.has_constraints:
            self.obstacle = params.env.obstacle
            self.obs_safety_thresh = 1 - params.beta_thresh # If more than self.obs_safety_thresh * n_parts paths intersect obstacle, blow up costs
                                     # for all paths for that member of the population
            
        self.alpha_thresh = params.alpha_thresh
        if params.gym_robotics:
            self.dO = params.env.observation_space.spaces['observation'].low.size
        else:
            self.dO = params.env.observation_space.shape[0]
        self.ac_ub, self.ac_lb = params.env.action_space.high, params.env.action_space.low
        self.dU = params.env.action_space.low.size
        self.ac_ub, self.ac_lb = params.env.action_space.high, params.env.action_space.low
        self.ac_ub = np.minimum(self.ac_ub, params.get("ac_ub", self.ac_ub))
        self.ac_lb = np.maximum(self.ac_lb, params.get("ac_lb", self.ac_lb))
        self.update_fns = params.get("update_fns", [])
        self.per = params.get("per", 1)

        self.model = get_required_argument(
            params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        )(params.prop_cfg.model_init_cfg)
        self.model_train_cfg = params.prop_cfg.get("model_train_cfg", {})
        self.prop_mode = get_required_argument(params.prop_cfg, "mode", "Must provide propagation method.")
        self.npart = get_required_argument(params.prop_cfg, "npart", "Must provide number of particles.")
        self.ign_var = params.prop_cfg.get("ign_var", False) or self.prop_mode == "E"

        self.obs_preproc = params.prop_cfg.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out)
        self.obs_postproc2 = params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs)
        self.targ_proc = params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs)

        self.opt_mode = get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.plan_hor = get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.obs_cost_fn = get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        self.ac_cost_fn = get_required_argument(params.opt_cfg, "ac_cost_fn", "Must provide cost on actions.")

        self.save_all_models = params.log_cfg.get("save_all_models", False)
        self.log_traj_preds = params.log_cfg.get("log_traj_preds", False)
        self.log_particles = params.log_cfg.get("log_particles", False)
        self.log_particles = True

        # Perform argument checks
        if self.prop_mode not in ["E", "DS", "MM", "TS1", "TSinf"]:
            raise ValueError("Invalid propagation method.")
        if self.prop_mode in ["TS1", "TSinf"] and self.npart % self.model.num_nets != 0:
            raise ValueError("Number of particles must be a multiple of the ensemble size.")
        if self.prop_mode == "E" and self.npart != 1:
            raise ValueError("Deterministic propagation methods only need one particle.")

        # Create action sequence optimizer
        opt_cfg = params.opt_cfg.get("cfg", {})
        self.optimizer = MPC.optimizers[params.opt_cfg.mode](
            sol_dim=self.plan_hor*self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            tf_session=None if not self.model.is_tf_model else self.model.sess,
            **opt_cfg
        )

        # Get value function
        self.value_func = params.value_func
        self.use_value = params.use_value
        self.stabilized_model = None

        # Controller state variables
        self.has_been_trained = params.prop_cfg.get("model_pretrained", False)
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])
        self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        )
        if self.model.is_tf_model:
            self.sy_cur_obs = tf.Variable(np.zeros(self.dO), dtype=tf.float32)
            self.cur_t_value = tf.Variable(3, dtype=tf.float32)
            self.ac_seq = tf.placeholder(shape=[1, self.plan_hor*self.dU], dtype=tf.float32)
            self.pred_cost, self.pred_traj = self._compile_cost(self.ac_seq, get_pred_trajs=True)
            self.optimizer.setup(self._compile_cost, True)
            self.model.sess.run(tf.variables_initializer([self.sy_cur_obs]))
        else:
            raise NotImplementedError()

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")
            
    def set_stabilized_model(self, model):
        """ Set the model used for stabilized_model"""
        self.stabilized_model = model

    def train(self, obs_trajs, acs_trajs, cost_trajs, stabilizable_obs):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            cost_trajs: A list of cost arrays.

        Returns: None.
        """
        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
            new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)
        
        # Train the model
        self.model.train(self.train_in, self.train_targs, **self.model_train_cfg)
        self.has_been_trained = True
        

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.optimizer.reset()
        if self.model.is_tf_model:
            for update_fn in self.update_fns:
                update_fn(self.model.sess)

    def act(self, obs, t, get_pred_cost=False, t_value=3):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        if not self.has_been_trained:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        if self.use_value:
            value = self.value_func.value(np.array([obs]), factored=True)
            value = np.mean(value[0]) + (t_value * np.std(value[0]))/np.sqrt(len(value[0]))

        if self.model.is_tf_model:
            self.sy_cur_obs.load(obs, self.model.sess)
            self.cur_t_value.load(t_value, self.model.sess)

        soln = self.optimizer.obtain_solution(self.prev_sol, self.init_var) # Getting action sequence
        self.prev_sol = np.concatenate([np.copy(soln)[self.per*self.dU:], np.zeros(self.per*self.dU)])
        self.ac_buf = soln[:self.per*self.dU].reshape(-1, self.dU)

        if get_pred_cost and not (self.log_traj_preds or self.log_particles):
            if self.model.is_tf_model:
                pred_cost = self.model.sess.run(
                    self.pred_cost,
                    feed_dict={self.ac_seq: soln[None]}
                )[0]
            else:
                raise NotImplementedError()
            return self.act(obs, t), pred_cost
        elif self.log_traj_preds or self.log_particles:
            pred_cost, pred_traj = self.model.sess.run(
                [self.pred_cost, self.pred_traj],
                feed_dict={self.ac_seq: soln[None]}
            )
            pred_cost, pred_traj = pred_cost[0], pred_traj[:, 0]
            if self.log_particles:
                self.pred_particles.append((pred_traj, pred_cost, obs))
            else:
                self.pred_means.append(np.mean(pred_traj, axis=1))
                self.pred_vars.append(np.mean(np.square(pred_traj - self.pred_means[-1]), axis=1))
            if get_pred_cost:
                return self.act(obs, t), pred_cost
        return self.act(obs, t)

    def dump_logs(self, primary_logdir, iter_logdir):
        """Saves logs to either a primary log directory or another iteration-specific directory.
        See __init__ documentation to see what is being logged.

        Arguments:
            primary_logdir (str): A directory path. This controller assumes that this directory
                does not change every iteration.
            iter_logdir (str): A directory path. This controller assumes that this directory
                changes every time dump_logs is called.

        Returns: None
        """
        self.model.save(iter_logdir if self.save_all_models else primary_logdir)
        if self.log_particles:
            savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
            self.pred_particles = []
        elif self.log_traj_preds:
            savemat(
                os.path.join(iter_logdir, "predictions.mat"),
                {"means": self.pred_means, "vars": self.pred_vars}
            )
            self.pred_means, self.pred_vars = [], []

    def _compile_cost(self, ac_seqs, get_pred_trajs=True):
        t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
        init_costs = tf.zeros([nopt, self.npart])
        ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU])
        ac_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_seqs, [1, 0, 2])[:, :, None],
            [1, 1, self.npart, 1]
        ), [self.plan_hor, -1, self.dU])
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])
        popsize = int(int(init_obs.shape[0])/self.npart)

        def unsafe(x):
            thresh = self.alpha_thresh
            dists = self.stabilized_model.kneighbors(x)[0]
            unsafe_idxs = np.where(dists > thresh)
            safe_idxs = np.where(dists <= thresh)
            dists[unsafe_idxs] = 1
            dists[safe_idxs] = 0
            return dists.reshape((-1, self.npart)).astype(bool)

        def continue_prediction(t, *args):
            return tf.less(t, self.plan_hor)

        if get_pred_trajs:
            pred_trajs = init_obs[None]

            def iteration(t, total_cost, cur_obs, pred_trajs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                next_obs = self.obs_postproc2(next_obs)
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
                return t + 1, total_cost + delta_cost, next_obs, pred_trajs


            _, costs, cur_obs, pred_trajs = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs, pred_trajs],
                shape_invariants=[
                    t.get_shape(), init_costs.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO])
                ]
            )
            
            if self.use_value:
                value_mean, value_var = self.value_func.value(cur_obs, factored=True)
                conf_mean, conf_var = tf.nn.moments(value_mean, axes=[0])
                value_UB = conf_mean + (self.cur_t_value/np.sqrt(self.model.num_nets) ) * tf.sqrt(conf_var)
                costs = costs + tf.reshape(value_UB, [-1, self.npart])
                costs = tf.where(tf.py_func(unsafe, [cur_obs], tf.bool), 1e6 * tf.ones_like(costs), costs)

            # Now check each pop size x nparts and see if it ever intersects obstacle, and 
            # blow up cost for all particles for that member of population
            pred_trajs = tf.reshape(pred_trajs, [self.plan_hor + 1, -1, self.npart, self.dO])
            
            if self.has_constraints:
                collisions = self.obstacle(pred_trajs)
                # Sum along the trajectory, so now will have sum > 0 for trajectories that collide
                collisions = tf.reduce_sum(collisions, axis=0)

                # Make collisions an indicator whether a specific trajectory was in collision
                collisions = tf.where(collisions > 0, tf.ones_like(collisions), collisions)

                pop_collisions = tf.reduce_sum(collisions, axis=1)

            # Taking mean over num particles here and replace nans with very high cost
            costs = tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)
            
            if self.has_constraints:
                # Now if total colliding TS's in member of pop > self.obs_safety_thresh % proportion of total TS,
                # blow up all costs for that member of the population:
                costs = tf.where(pop_collisions > int(self.obs_safety_thresh * self.npart), 1e6 * tf.ones_like(costs), costs)

            return costs, pred_trajs
        else:
            def iteration(t, total_cost, cur_obs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                return t + 1, total_cost + delta_cost, self.obs_postproc2(next_obs)

            _, costs, cur_obs = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs]
            )
            if self.use_value:
                value_mean, value_var = self.value_func.value(cur_obs, factored=True)
                conf_mean, conf_var = tf.nn.moments(value_mean, axes=[0])
                value_UB = conf_mean + (self.cur_t_value/np.sqrt(self.model.num_nets) ) * tf.sqrt(conf_var)
                costs = costs + tf.reshape(value_UB, [-1, self.npart])
                costs = tf.where(tf.reshape(tf.py_func(unsafe, [cur_obs], tf.bool), costs.get_shape()), 1e6 * tf.ones_like(costs), costs)

            # Replace nan costs with very high costand take mean over the particles
            costs = tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)
            return costs

    def _predict_next_obs(self, obs, acs):
        proc_obs = self.obs_preproc(obs)

        if self.model.is_tf_model:
            # TS Optimization: Expand so that particles are only passed through one of the networks.
            if self.prop_mode == "TS1":
                proc_obs = tf.reshape(proc_obs, [-1, self.npart, proc_obs.get_shape()[-1]])
                sort_idxs = tf.nn.top_k(
                    tf.random_uniform([tf.shape(proc_obs)[0], self.npart]),
                    k=self.npart
                ).indices
                tmp = tf.tile(tf.range(tf.shape(proc_obs)[0])[:, None], [1, self.npart])[:, :, None]
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                proc_obs = tf.gather_nd(proc_obs, idxs)
                proc_obs = tf.reshape(proc_obs, [-1, proc_obs.get_shape()[-1]])
            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                proc_obs, acs = self._expand_to_ts_format(proc_obs), self._expand_to_ts_format(acs)

            # Obtain model predictions
            inputs = tf.concat([proc_obs, acs], axis=-1)
            mean, var = self.model.create_prediction_tensors(inputs)
            if self.model.is_probabilistic and not self.ign_var:
                predictions = mean + tf.random_normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)
                if self.prop_mode == "MM":
                    model_out_dim = predictions.get_shape()[-1].value

                    predictions = tf.reshape(predictions, [-1, self.npart, model_out_dim])
                    prediction_mean = tf.reduce_mean(predictions, axis=1, keep_dims=True)
                    prediction_var = tf.reduce_mean(tf.square(predictions - prediction_mean), axis=1, keep_dims=True)
                    z = tf.random_normal(shape=tf.shape(predictions), mean=0, stddev=1)
                    samples = prediction_mean + z * tf.sqrt(prediction_var)
                    predictions = tf.reshape(samples, [-1, model_out_dim])
            else:
                predictions = mean

            # TS Optimization: Remove additional dimension
            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                predictions = self._flatten_to_matrix(predictions)
            if self.prop_mode == "TS1":
                predictions = tf.reshape(predictions, [-1, self.npart, predictions.get_shape()[-1]])
                sort_idxs = tf.nn.top_k(
                    -sort_idxs,
                    k=self.npart
                ).indices
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                predictions = tf.gather_nd(predictions, idxs)
                predictions = tf.reshape(predictions, [-1, predictions.get_shape()[-1]])

            return self.obs_postproc(obs, predictions)
        else:
            raise NotImplementedError()

    def _expand_to_ts_format(self, mat):
        # import IPython; IPython.embed()
        dim = mat.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(mat, [-1, self.model.num_nets, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [self.model.num_nets, -1, dim]
        )

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(ts_fmt_arr, [self.model.num_nets, -1, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [-1, dim]
        )
