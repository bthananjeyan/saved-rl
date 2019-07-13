from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


TARGET = np.array([0.13345871, 0.21923056, -0.10861196])
THRESH = 0.05
HORIZON = 100
FAILURE_COST = 0

class ReacherSparse3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.viewer, self.time = None, 0
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.goal = np.copy(TARGET)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dir_path, 'assets/reacher3d.xml'), 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        self.time += 1
        ob = self._get_obs()
        cost = (np.sum(np.square(self.get_EE_pos(ob[None]) - self.goal)) > THRESH).astype(np.float32)
        done = HORIZON <= self.time
        if done and not self.is_stable(ob):
            cost += FAILURE_COST
        return ob, cost, done, dict(cost_dist=0, cost_ctrl=0)

    def process_action(self, state, action):
        return action


    def post_process(self, obs, acs, costs):
        ob_costs = np.array([np.sum(np.square(self.get_EE_pos(ob[None]) - self.goal)) for ob in obs])
        return (ob_costs[:-1] > THRESH).astype(np.float32)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 270

    def reset_model(self):
        qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
        qpos[-3:] += np.random.normal(loc=0, scale=0.1, size=[3])
        qvel[-3:] = 0
        self.time = 0
        # self.goal = qpos[-3:]
        qpos[-3:] = self.goal = np.copy(TARGET)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat[:-3],
        ])

    def get_EE_pos(self, states):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], states[:, 4:5], states[:, 5:6], states[:, 6:]

        rot_axis = np.concatenate([np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), -np.sin(theta2)],
                                  axis=1)
        rot_perp_axis = np.concatenate([-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
        cur_end = np.concatenate([
            0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
            0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
            -0.4 * np.sin(theta2)
        ], axis=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
                rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end

    def is_stable(self, ob):
        return (np.sum(np.square(self.get_EE_pos(ob[None]) - self.goal)) < THRESH).astype(bool)
