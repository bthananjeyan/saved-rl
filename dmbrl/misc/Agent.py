from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from dmbrl.misc.video_recorder import VideoRecorder
from dotmap import DotMap

import time


class Agent:
    """An general class for RL agents.
    """
    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        self.env = params.env
        self.noise_stddev = params.noise_stddev if params.get("noisy_actions", False) else None

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")
        if (not isinstance(self.noise_stddev, float)) and params.get("noisy_actions", False):
            raise ValueError("Must provide standard deviation for noise for noisy actions.")

        if self.noise_stddev is not None:
            self.dU = self.env.action_space.shape[0]

    def sample(self, horizon, policy, record_fname=None, t_value=3):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'cost_sum'.
        """
        def process_obs(obs, gym_robotics=False):
            if gym_robotics:
                return obs['observation']
            else:
                return obs
        video_record = record_fname is not None
        recorder = None if not video_record else VideoRecorder(self.env, record_fname)

        times, costs = [], []
        initial_obs = self.env.reset()
        O, A, cost_sum, done = [process_obs(initial_obs, self.env.gym_robotics)], [], 0, False
        
        policy.reset()
        for t in range(horizon):
            if video_record:
                recorder.capture_frame()
            start = time.time()
            A.append(policy.act(O[t], t, t_value=t_value))
            times.append(time.time() - start)

            if self.noise_stddev is None:
                action = A[t]
                action = self.env.process_action(O[t], action)
                obs, cost, done, info = self.env.step(action)
            else:
                action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
                action = self.env.process_action(O[t], action)
                action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
                obs, cost, done, info = self.env.step(action)
            O.append(process_obs(obs, self.env.gym_robotics))
            cost_sum += cost
            costs.append(cost)
            if done:
                break

        if video_record:
            recorder.capture_frame()
            recorder.close()

        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        costs = np.array(costs)
        values = np.cumsum(costs[::-1])[::-1]
        
        if self.env.is_stable(process_obs(obs, self.env.gym_robotics)):
            stabilizable_obs = O
            is_stable = True
        else:
            stabilizable_obs = []
            is_stable = False

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "cost_sum": cost_sum,
            "costs": costs,
            "values": values,
            "stabilizable_obs" : stabilizable_obs,
            "is_stable": is_stable
        }
