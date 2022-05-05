# Third party imports
import numpy as np

# Acceleration parameters
ACCELERATION = [-1, 0, 1]
ACTIONS = [[x_col_acc, y_row_acc] for x_col_acc in [-1, 0, 1] for y_row_acc in [-1, 0, 1]]
P_ACC_SUCCEED = 0.8
P_ACC_FAIL = 1 - P_ACC_SUCCEED

# Velocity parameters
MAX_VELOCITY = 5
MIN_VELOCITY = -5
VEL_MIN = -5
VEL_MAX = 5
VELOCITIES = np.arange(VEL_MIN, VEL_MAX + 1)

# Learning parameters
GAMMA = 0.9
ETA = 0.1
EPSILON = 0.2
INIT_TEMP = 10
TEMP_DISSIPATION_FRAC = 0.9999
VALUE_ITERATION_TERMINATION_THRESHOLD = 0
INIT_VAL = -500000

N_CORES = 6