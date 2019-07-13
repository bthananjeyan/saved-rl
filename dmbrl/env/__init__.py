from gym.envs.registration import register


register(
	id='PointBot-v0',
	entry_point='dmbrl.env.pointbot:PointBot')

register(
    id='MBRLReacherSparse3D-v0',
    entry_point='dmbrl.env.reachersparse:ReacherSparse3DEnv')

register(
    id='MBRL-PickAndPlace-v1',
    entry_point='dmbrl.env.pick_and_place:FetchPickAndPlaceEnv'
)
