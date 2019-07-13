import gym
import numpy as np

from dmbrl.env.pick_and_place import FetchPickAndPlaceEnv
import pickle


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
simStates = []
objStates = []
infos = []

noise_var = 0.75
gain = 5.5


def main():
    env = FetchPickAndPlaceEnv()
    print("SIM")
    print(env.sim)
    assert(env.sim)
    
    numItr = 50
    initStateSpace = "random"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)


    fileName = "experts/fetchpickandplace/data_fetch"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    # fileName += ".npz"
    fileName += ".p"

    print(np.array(observations).shape)


    # np.savez_compressed(fileName, acs=actions, obs=observations, info=infos, sim_states=simStates, obj_states=objStates) # save the file
    pickle.dump({'acs' : actions, 'obs' : observations, 'info' : infos, 'sim_states' : simStates, 'obj_states' : objStates}, open(fileName, "wb"))

def goToGoal(env, lastObs):

    goal = lastObs['desired_goal']
    object_rel_pos = lastObs['observation'][:3]
    episodeAcs = []
    episodeObs = []
    episodeSimStates = []
    episodeObjStates = []
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)
    episodeSimStates.append(env.sim.get_state())
    episodeObjStates.append(env.sim.data.get_joint_qpos('object0:joint').copy())

    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep < 50:
        # env.render()
        action = [0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]*(gain + np.random.randn() * noise_var)
        action[len(action)-1] = 0.05 #open

        obsDataNew, reward, done, info = env.step(action)
        print(np.linalg.norm(object_rel_pos), obsDataNew['observation'][3:5].sum(), np.linalg.norm(obsDataNew['observation'][5:8]), reward)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeSimStates.append(env.sim.get_state())
        episodeObjStates.append(env.sim.data.get_joint_qpos('object0:joint').copy())

        object_rel_pos = obsDataNew['observation'][:3]

    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep < 50:
        # env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*(gain + np.random.randn() * noise_var)

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        # print(obsDataNew['observation'])
        print(np.linalg.norm(object_rel_pos), obsDataNew['observation'][3:5].sum(), np.linalg.norm(obsDataNew['observation'][5:8]), reward)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeSimStates.append(env.sim.get_state())
        episodeObjStates.append(env.sim.data.get_joint_qpos('object0:joint').copy())

        object_rel_pos = obsDataNew['observation'][:3]

    goal_rel_pos = obsDataNew['observation'][5:8]
    while np.linalg.norm(goal_rel_pos) >= 0.01 and timeStep < 50:
        # env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal_rel_pos)):
            action[i] = (goal_rel_pos)[i]*(gain + np.random.randn() * noise_var)

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1
        print(np.linalg.norm(object_rel_pos), obsDataNew['observation'][3:5].sum(), np.linalg.norm(obsDataNew['observation'][5:8]), reward)

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeSimStates.append(env.sim.get_state())
        episodeObjStates.append(env.sim.data.get_joint_qpos('object0:joint').copy())


        goal_rel_pos = obsDataNew['observation'][5:8]

    while True and timeStep < 50: #limit the number of timesteps in the episode to a fixed duration
        # env.render()
        action = [0, 0, 0, 0]
        action[len(action)-1] = -0.005 # keep the gripper closed
        for i in range(3):
            action[i] += np.random.randn() * 0.01

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        episodeSimStates.append(env.sim.get_state())
        episodeObjStates.append(env.sim.data.get_joint_qpos('object0:joint').copy())

        object_rel_pos = obsDataNew['observation'][:3]

        if timeStep >= 50: break

    assert len(episodeObs) == 51, (len(episodeObs), timeStep)


    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)
    
    simStates.append(episodeSimStates)
    objStates.append(episodeObjStates)


if __name__ == "__main__":
    main()
