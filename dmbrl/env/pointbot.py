"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise and air resistance
proportional to its velocity. State representation is (x, vx, y, vy). Action
representation is (fx, fy), and mass is assumed to be 1.
"""

import os
import pickle

import os.path as osp
import numpy as np
import tensorflow as tf
from gym import Env
from gym import utils
from gym.spaces import Box

from .pointbot_const import *

def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

def lqr_gains(A, B, Q, R, T):
    Ps = [Q]
    Ks = []
    for t in range(T):
        P = Ps[-1]
        Ps.append(Q + A.T.dot(P).dot(A) - A.T.dot(P).dot(B)
            .dot(np.linalg.inv(R + B.T.dot(P).dot(B))).dot(B.T).dot(P).dot(A))
    Ps.reverse()
    for t in range(T):
        Ks.append(-np.linalg.inv(R + B.T.dot(Ps[t+1]).dot(B)).dot(B.T).dot(P).dot(A))
    return Ks, Ps


class PointBot(Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.A = np.eye(4)
        self.A[2,3] = self.A[0,1] = 1
        self.A[1,1] = self.A[3,3] = 1 - AIR_RESIST
        self.B = np.array([[0,0], [1,0], [0,0], [0,1]])
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(2) * MAX_FORCE, np.ones(2) * MAX_FORCE)
        self.observation_space = Box(-np.ones(4) * np.float('inf'), np.ones(4) * np.float('inf'))
        # self.obstacle = ComplexObstacle([[-30, -20], [-20, 20]])
        self.start_state = START_STATE

    def set_mode(self, mode):
        self.mode = mode
        self.obstacle = OBSTACLE[mode]
        if self.mode == 1:
            self.start_state = [-100, 0, 0, 0]

    def process_action(self, state, action):
        return action

    def step(self, a):
        a = process_action(a)
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        if not self.obstacle(self.state):
            self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = HORIZON <= self.time
        if self.done and not self.is_stable(self.state):
            self.cost[-1] += FAILURE_COST
            cur_cost += FAILURE_COST
        return self.state, cur_cost, self.done, {}

    def reset(self):
        self.state = self.start_state + np.random.randn(4)
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _next_state(self, s, a):
        return self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(len(s))

    def step_cost(self, s, a):
        if HARD_MODE:
            return int(np.linalg.norm(np.subtract(GOAL_STATE, s)) > GOAL_THRESH) + self.obstacle(s)
        return np.linalg.norm(np.subtract(GOAL_STATE, s))

    def collision_cost(self, obs):
        return self.obstacle(obs)


    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        return np.random.random(2) * 2 * MAX_FORCE - MAX_FORCE

    def plot_trajectory(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        plt.scatter(states[:,0], states[:,2])
        plt.show()

    # Returns whether a state is stable or not
    def is_stable(self, s):
        return np.linalg.norm(np.subtract(GOAL_STATE, s)) <= GOAL_THRESH

    def teacher(self, sess=None):
        return PointBotTeacher()

class PointBotTeacher(object):

    def __init__(self):
        self.env = PointBot()
        self.Ks, self.Ps = lqr_gains(self.env.A, self.env.B, np.eye(4), 50 * np.eye(2), HORIZON)
        self.demonstrations = []
        self.outdir = "data/pointbot"

    def get_rollout(self):
        obs = self.env.reset()
        O, A, cost_sum, costs = [obs], [], 0, []
        noise_std = 0.2
        for i in range(HORIZON):
            if self.env.mode == 1:
                noise_idx = np.random.randint(int(HORIZON * 2 / 3))
                if i < HORIZON / 2:
                    action = [0.1, 0.1]
                else:
                    action = self._expert_control(obs, i)
            else:
                noise_idx = np.random.randint(int(HORIZON))
                if i < HORIZON / 4:
                    action = [0.1, 0.25]
                elif i < HORIZON / 2:
                    action = [0.4, 0.]
                elif i < HORIZON / 3 * 2:
                    action = [0, -0.5]
                else:
                    action = self._expert_control(obs, i)

            if i < noise_idx:
                action = (np.array(action) +  np.random.normal(0, noise_std, self.env.action_space.shape[0])).tolist()

            A.append(action)
            obs, cost, done, info = self.env.step(action)
            O.append(obs)
            cost_sum += cost
            costs.append(cost)
            if done:
                break
        costs = np.array(costs)

        values = np.cumsum(costs[::-1])[::-1]
        if self.env.is_stable(obs):
            stabilizable_obs = O
        else:
            stabilizable_obs = []
            return self.get_rollout()

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "cost_sum": cost_sum,
            "costs": costs,
            "values": values,
            "stabilizable_obs" : stabilizable_obs
        }

    def _get_gain(self, t):
        return self.Ks[t]

    def _expert_control(self, s, t):
        return self._get_gain(t).dot(s)
