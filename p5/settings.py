from pathlib import Path
import numpy as np

ACCELERATION = [-1, 0, 1]
ACTIONS = [[x_col_acc, y_row_acc] for x_col_acc in [-1, 0, 1] for y_row_acc in [-1, 0, 1]]
MAX_VELOCITY = 5
MIN_VELOCITY = -5
INIT_POS = (0, 0)
INIT_VEL = (0, 0)
INIT_ACC = (0, 0)
INIT_T = 0
VEL_MIN = -5
VEL_MAX = 5
VELOCITIES = np.arange(VEL_MIN, VEL_MAX + 1)
GAMMA = 0.9
P_ACC_SUCCEED = 0.8
P_ACC_FAIL = 1 - P_ACC_SUCCEED
VALUE_ITERATION_TERMINATION_THRESHOLD = 0
N_CORES = 6
