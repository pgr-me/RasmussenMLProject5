#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, utils.py

This module provides miscellaneous utility functions.

"""
# Third party libraries
import numpy as np

# Local imports
from p5.settings import VEL_MAX, VEL_MIN, P_ACC_FAIL


def bresenham(x_col_0: int, y_row_0: int, x_col_1: int, y_row_1: int):
    """
    Compute the cells visited between two raster cells using the Breseham algorithm.
    :param x_col_0: Starting x
    :param y_row_0: Starting y
    :param x_col_1: Ending x
    :param y_row_1: Ending y
    First element of returned array is the starting point.
    Source: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm#:~:text=Bresenham%27s%20line%20algorithm%20is%20a%20line%20drawing%20algorithm,approximation%20to%20a%20straight%20line%20between%20two%20points.
    TinyURL: https://tinyurl.com/2p9692h6
    """
    dx = abs(x_col_1 - x_col_0)
    sx = 1 if x_col_0 < x_col_1 else -1
    dy = -abs(y_row_1 - y_row_0)
    sy = 1 if y_row_0 < y_row_1 else -1
    error = dx + dy
    output = []
    while True:
        output.append((x_col_0, y_row_0))
        if (x_col_0 == x_col_1) & (y_row_0 == y_row_1):
            return output
        e2 = 2 * error
        if e2 >= dy:
            if x_col_0 == x_col_1:
                return output
            error += dy
            x_col_0 += sx
        if e2 <= dx:
            if y_row_0 == y_row_1:
                return output
            error += dx
            y_row_0 += sy


def compute_velocity(vel: tuple, acc: tuple) -> tuple:
    """
    Update car speed based on acceleration.
    :param vel: x-y velocity pair
    :param acc: x-y acceleration pair
    :return: x-y pair of velocities
    """
    x_col_vel, y_row_vel = vel
    x_col_acc, y_row_acc = acc
    x_col_vel_1 = x_col_vel + x_col_acc
    y_row_vel_1 = y_row_vel + y_row_acc

    # Update velocities subject to max and min velocity limits
    x_col_vel_1 = VEL_MAX if x_col_vel_1 > VEL_MAX else x_col_vel_1
    x_col_vel_1 = VEL_MIN if x_col_vel_1 < VEL_MIN else x_col_vel_1
    y_row_vel_1 = VEL_MAX if y_row_vel_1 > VEL_MAX else y_row_vel_1
    y_row_vel_1 = VEL_MIN if y_row_vel_1 < VEL_MIN else y_row_vel_1

    return x_col_vel_1, y_row_vel_1


def minkowski_distance(x, y, p=2):
    """
    Compute the Minkowski distance between a vector and a vector or matrix.
    :param x: x vector
    :param y: y vector or matrix
    :param p: p-norm
    p-norm = 1 for Manhattan distance, 2 for Euclidean distance, etc.
    From https://github.com/pgr-me/RasmussenMLProject2/blob/main/p2/algorithms/utils.py
    ^^That's me
    """
    diff = np.abs(x - y).T
    power = np.power(diff.T, p * np.ones(len(diff))).T
    power_sum = np.sum(power, axis=0)
    return np.power(power_sum.T, 1 / p)


def minkowski_distances(X: np.array, Y: np.array, p: int = 2) -> np.array:
    """
    Compute Minkowski distances between two matrices.
    :param X: Array of points where each row is a point and each column is a dimension
    :param Y: Array of points where each row is a point and each column is a dimension
    :param p: p-norm
    :return: Array of distances between each observation in X and Y
    Output includes same-point distances (e.g., point 1 distance from point 1 distance is computed).
    Output ordered s.t. point 1 (P1) repeated across points 1 to m, P2 repeated across points 1 to m, etc.
    p-norm = 1 for Manhattan distance, 2 for Euclidean distance, etc.
    Example output:
        array([0, 6.38634802, 8.75831689, ..., 7.43018813, 6.41678194])
    From https://github.com/pgr-me/RasmussenMLProject2/blob/main/p2/algorithms/utils.py
    ^^That's me
    """
    row_repeat = np.repeat(X, len(Y), axis=0)
    stacked = np.vstack([Y for y in range(len(X))])
    diff = np.abs(np.subtract(row_repeat, stacked))
    power = np.power(diff, p * np.ones(diff.shape))
    power_sum = np.sum(power, axis=1)
    return np.power(power_sum, np.ones(power_sum.shape) / p)


def realize_action(acc: tuple) -> tuple:
    """
    Get resulting acceleration.
    :param acc: Chosen x-y acceleration pair
    :return: Resulting x-y acceleration pair
    """
    x_col_acc, y_row_acc = acc
    if np.random.random() > P_ACC_FAIL:
        x_col_acc, y_row_acc = x_col_acc, y_row_acc
    else:
        x_col_acc, y_row_acc = 0, 0
    return x_col_acc, y_row_acc
