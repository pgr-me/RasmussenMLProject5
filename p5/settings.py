#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, settings.py

This module provides constants used by the program.

"""
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
VELOCITIES = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]

# Learning parameters
GAMMA = 0.9
ETA = 0.1
EPSILON = 0.2
INIT_TEMP = 10
MIN_TEMP = 1
TEMP_DISSIPATION_FRAC = 0.99999
VALUE_ITERATION_TERMINATION_THRESHOLD = 0.01
INIT_VAL = -100
FRAC_UNVISITED = 0.1

N_CORES = 4
OOB_PENALTIES = ["stay-in-place", "back-to-beginning"]