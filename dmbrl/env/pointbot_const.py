from .obstacle import Obstacle, ComplexObstacle

"""
Constants associated with the PointBot env.
"""

START_POS = [-50, 0]
END_POS = [0, 0]
GOAL_THRESH = 1.
START_STATE = [START_POS[0], 0, START_POS[1], 0]
GOAL_STATE = [END_POS[0], 0, END_POS[1], 0]

MAX_FORCE = 1
HORIZON = 100

NOISE_SCALE = 0.05
AIR_RESIST = 0.2

HARD_MODE = True
FAILURE_COST = 0


OBSTACLE = {
	1: ComplexObstacle([[[-1000, -999], [-1000, -999]]]),
	2: ComplexObstacle([[[-30, -20], [-20, 20]]]),
	3: ComplexObstacle([[[-30, -20], [-20, -10]], [[-30, -20], [0, 20]]]),
	4: ComplexObstacle([[[-30, -20], [-20, 20]], [[-20, 5], [10, 20]], [[0, 5], [5, 10]], [[-20, 5], [-20, -10]]])
}
