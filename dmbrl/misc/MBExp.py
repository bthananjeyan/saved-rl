from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
from copy import deepcopy
from time import time, localtime, strftime

import numpy as np
from scipy.io import savemat, loadmat
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent
from dmbrl.modeling.utils import LinearSchedule
from dmbrl.modeling.utils import InterpolatingLinearSchedule
from sklearn.neighbors import NearestNeighbors as knn

def load_teacher_samples_gym(demo_load_path, env, lower, upper, max_num_samples):
    mat = loadmat(demo_load_path)
    samples = []
    for i, acs in enumerate(mat['actions']):
        cur = {}
        cur['ac'] = acs
        cur['obs'] = mat['observations'][i]
        cur['costs'] = env.post_process(mat['observations'][i], acs, mat['rewards'][i])
        cur['values'] = np.cumsum(cur['costs'][::-1])[::-1]
     
        # Check if total cost is in reasonable range and that the goal was achieved at the end
        if cur['values'][0] < upper and cur['values'][0] > lower and cur['costs'][-1] == 0:
            cur['stabilizable_obs'] = mat['observations'][i]
            if len(samples) < max_num_samples:
                samples.append(cur)
    return samples

def load_teacher_samples_gym_robotics(demo_load_path, env, lower=0, upper=100, max_num_samples=100):
    data = pickle.load( open(demo_load_path, "rb") )
    samples = []
  
    actions = data['acs']
    sim_states = data['sim_states']
    obj_states = data['obj_states']
    obs_data = np.array(data['obs'])

    observations = np.zeros((obs_data.shape[0], obs_data.shape[1], obs_data[0][0]['observation'].shape[0]))
    achieved_goals = np.zeros((obs_data.shape[0], obs_data.shape[1], obs_data[0][0]['achieved_goal'].shape[0]))
    desired_goals = np.zeros((obs_data.shape[0], obs_data.shape[1], obs_data[0][0]['desired_goal'].shape[0]))
    
    costs = deepcopy(data['info'])
    
    for i in range(len(obs_data)):
        for j in range(len(obs_data[i])):
            observations[i][j] = np.array(obs_data[i][j]['observation'])
            achieved_goals[i][j] = np.array(obs_data[i][j]['achieved_goal'])
            desired_goals[i][j] = np.array(obs_data[i][j]['desired_goal'])

    for i in range(len(data['info'])):
        for j in range(len(data['info'][i])):
            costs[i][j] = 1 - data['info'][i][j]['is_success']

    for i, acs in enumerate(actions):
        cur = {}
        cur['ac'] = acs
        cur['obs'] = observations[i]
        cur['sim_states'] = sim_states[i]
        cur['obj_states'] = obj_states[i]
        cur['costs'] = env.post_process(observations[i], acs, costs[i])
        cur['values'] = np.cumsum(cur['costs'][::-1])[::-1]

        # Check if total cost is in reasonable range and that the goal was achieved at the end
        if cur['values'][0] < upper and cur['values'][0] > lower and cur['costs'][-1] == 0:
            cur['stabilizable_obs'] = observations[i]
            if len(samples) < max_num_samples:
                samples.append(cur) 
                
    return samples

class MBExperiment:
    def __init__(self, params):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.
                    .demo_low_cost (int): Minimum allowed cost for demonstrations
                    .demo_high_cost (int): Maximum allowed cost for demonstrations
                    .num_demos (int): Number of demonstrations
                    .ss_buffer_size (int): Size of buffer of safe states that density model is
                        trained on
                    .gym_robotics (bool): Indicates whether env is a gym robotics env, in which
                        case there are some small differences in data loading and environment
                        parameters
                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """
        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.demo_low_cost = params.exp_cfg.demo_low_cost
        self.demo_high_cost = params.exp_cfg.demo_high_cost
        self.num_demos = params.exp_cfg.num_demos
        self.ss_buffer_size = params.exp_cfg.ss_buffer_size
        self.gym_robotics = params.exp_cfg.gym_robotics
        
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        if params.sim_cfg.get("stochastic", False):
            self.agent = Agent(DotMap(
                env=self.env, noisy_actions=True,
                noise_stddev=get_required_argument(
                    params.sim_cfg,
                    "noise_std",
                    "Must provide noise standard deviation in the case of a stochastic environment."
                )
            ))
        else:
            self.agent = Agent(DotMap(env=self.env, noisy_actions=False))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")
        self.value = get_required_argument(params.exp_cfg, "value", "Must provide a value function.")
        self.target = get_required_argument(params.exp_cfg, "value_target", "Must provide a value function.")
        self.value.target = self.target

        self.logdir = os.path.join(
            get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."),
            strftime("%Y-%m-%d--%H:%M:%S", localtime())
        )
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)
        self.load_samples = params.exp_cfg.get("load_samples", True)
        self.demo_load_path = params.exp_cfg.get("demo_load_path", None)
        self.use_value = params.exp_cfg.get("use_value", True)
        self.teacher = params.exp_cfg.get("teacher")
        self.stabilizable_observations = []
        self.tvalue_schedule = LinearSchedule(3, 3, 500)
        self.stabilized_model = knn(n_neighbors=1)
        self.target_update_freq = 1

    def run_experiment(self):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        traj_obs, traj_acs, traj_rets, traj_costs = [], [], [], []

        # Perform initial rollouts
        samples = []
        if self.load_samples:
            if not self.gym_robotics:
                samples = load_teacher_samples_gym(self.demo_load_path, self.env, self.demo_low_cost, self.demo_high_cost, self.num_demos)
            else:
                samples = load_teacher_samples_gym_robotics(self.demo_load_path, self.env, self.demo_low_cost, self.demo_high_cost, self.num_demos)

            for i in range(len(samples)):
                traj_obs.append(samples[i]["obs"])
                traj_acs.append(samples[i]["ac"])
                traj_costs.append(samples[i]["costs"])
                self.stabilizable_observations.extend(samples[i]["stabilizable_obs"])
        else:
            for i in range(self.num_demos):
                s = self.teacher.get_rollout()
                samples.append(s)
                traj_obs.append(samples[-1]["obs"])
                traj_acs.append(samples[-1]["ac"])
                traj_costs.append(samples[-1]["costs"])
                self.stabilizable_observations.extend(samples[-1]["stabilizable_obs"])       

        # Fit density model to demonstrations           
        if self.stabilized_model is not None:
            self.stabilized_model.fit(np.array(self.stabilizable_observations))
        else:
            self.stabilized_model=None
        self.policy.set_stabilized_model(self.stabilized_model)
        
        if self.num_demos > 0:
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["costs"] for sample in samples],
                np.array(self.stabilizable_observations)
            )
            if self.use_value:
                # Train value function using teacher rollouts
                self.value.train(
                    [sample["obs"][:-1] for sample in samples],
                    [sample["costs"] for sample in samples],
                    [sample["obs"][1:] for sample in samples],
                    [sample["values"] for sample in samples],
                    use_TD=False,
                    terminal_states=[sample["obs"][-1] for sample in samples],
                    copy_target=True
                )

        demo_samples = deepcopy(samples)

        # Training loop
        for i in range(self.ntrain_iters):
            print("####################################################################")
            print("Starting training iteration %d." % (i + 1))

            iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)


            samples = []
            for j in range(self.nrecord):
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy,
                        os.path.join(iter_dir, "rollout%d.mp4" % j)
                    )
                )
            if self.nrecord > 0:
                for item in filter(lambda f: f.endswith(".json"), os.listdir(iter_dir)):
                    os.remove(os.path.join(iter_dir, item))
            for j in range(max(self.neval, self.nrollouts_per_iter) - self.nrecord):
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy
                    )
                )
            print("Costs obtained:", [sample["cost_sum"] for sample in samples[:self.neval]])
            traj_obs.extend([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
            traj_acs.extend([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
            traj_rets.extend([sample["cost_sum"] for sample in samples[:self.neval]])
            traj_costs.extend([sample["costs"] for sample in samples[:self.nrollouts_per_iter]])

            samples = samples[:self.nrollouts_per_iter]

            self.policy.dump_logs(self.logdir, iter_dir)
            if self.use_value:
                self.value.dump_logs(self.logdir, iter_dir)

            savemat(
                os.path.join(self.logdir, "logs.mat"),
                {
                    "observations": traj_obs,
                    "actions": traj_acs,
                    "returns": traj_rets,
                    "costs": traj_costs
                }
            )
            # Delete iteration directory if not used
            if len(os.listdir(iter_dir)) == 0:
                os.rmdir(iter_dir)

            if i < self.ntrain_iters - 1:
                self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["costs"] for sample in samples],
                    np.array(self.stabilizable_observations)
                )

                if self.gym_robotics:
                    current_stabilizable_obs = np.array([sample["stabilizable_obs"] for sample in samples]).reshape((-1,  self.env.observation_space.spaces['observation'].low.size))
                else:
                    current_stabilizable_obs = np.array([sample["stabilizable_obs"] for sample in samples]).reshape((-1,  self.env.observation_space.shape[0]))
                if self.use_value:
                    copy_target = i % self.target_update_freq == 0
                    # Train value function using teacher rollouts
                    self.value.train(
                        [sample["obs"][:-1] for sample in samples],
                        [sample["costs"] for sample in samples],
                        [sample["obs"][1:] for sample in samples],
                        [sample["values"] for sample in samples],
                        use_TD=True,
                        terminal_states=[sample["obs"][-1] for sample in samples],
                        copy_target=copy_target
                    )


                if len(current_stabilizable_obs):
                    current_stabilizable_obs = [c for c in current_stabilizable_obs]
                    self.stabilizable_observations.extend(current_stabilizable_obs)
                    self.stabilizable_observations = self.stabilizable_observations[-self.ss_buffer_size:]

                if self.stabilized_model is not None:
                    self.stabilized_model.fit(np.array(self.stabilizable_observations))
                    self.policy.set_stabilized_model(self.stabilized_model)
                    pickle.dump(self.stabilized_model, open(os.path.join(self.logdir, "stabilized_model.pkl"), "wb"))