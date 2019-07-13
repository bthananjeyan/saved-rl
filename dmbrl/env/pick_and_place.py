import os.path as osp

from gym import utils
import numpy as np
from gym.envs.robotics import rotations, robot_env
import gym.envs.robotics.utils as roboutils


from dmbrl.env import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = osp.join(osp.dirname(__file__), 'assets/fetch', 'pick_and_place.xml')

class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0., target_range=0., distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def post_process(self, obs, acs, costs):
        c = np.zeros(len(obs))
        for i in range(len(c)):
            c[i] = self.compute_reward(obs[i], None, None)
        return costs

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + np.array([0.1, 0.1])
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = roboutils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        goal_rel_pos = self.goal - object_pos
        obs = np.concatenate([
            object_rel_pos.ravel(), gripper_state, goal_rel_pos.ravel()
        ])


        return {
            'observation': obs.copy(),
            'achieved_goal': obs.copy(),
            'desired_goal': np.zeros(3),
        }

    def set_start_state(self, sim_state, obj_state):
        self.sim.set_state(sim_state)
        self.sim.data.set_joint_qpos('object0:joint', obj_state)
        self.sim.forward()

    def process_action(self, state, action):
        action = np.copy(action)
        if np.linalg.norm(state[:3]) < 0.035:
            action[-1] = -0.05
        else:
            action[-1] = 0.05
        return action

    def get_sim_state(self):
        return self.sim.get_state()

    def get_obj_state(self):
        return self.sim.data.get_joint_qpos('object0:joint')

    def is_stable(self, ob):
        d = np.linalg.norm(ob[:3])
        grip = ob[3:5].sum()
        d2 = np.linalg.norm(ob[5:8])
        return (d2 < self.distance_threshold).astype(bool)
