#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, q_learning.py

This module provides the functions used for Q-learning.

"""
# Third party imports
import numpy as np
import pandas as pd

# Local imports
from p5.track import Track
from p5.settings import EPSILON
from p5.utils import bresenham


def compute_position(pos_0: tuple, vel_1: tuple, track: Track) -> tuple:
    """
    Update position based on velocity.
    :param pos_0: Initial x-y position
    :param vel_1: x-y velocity
    :param track: Track object
    :return: x-y position
    """
    x_col_pos_0, y_row_pos_0 = pos_0
    x_col_vel_1, y_row_vel_1 = vel_1
    x_col_pos_1 = x_col_pos_0 + x_col_vel_1
    y_row_pos_1 = y_row_pos_0 + y_row_vel_1

    path = bresenham(x_col_pos_0, y_row_pos_0, x_col_pos_1, y_row_pos_1)
    for pos in path:

        # Case when car hits a wall
        if pos in track.wall_cells:
            # Case when car remains in its current position
            if track.oob_penalty == "stay-in-place":
                x_col_pos_1 = x_col_pos_0
                y_row_pos_1 = y_row_pos_0
            # Case when car returns to start
            else:
                start_pos_ix = np.random.choice(range(len(track.start_cells)))
                start_pos = list(track.start_cells)[start_pos_ix]
                x_col_pos_1, y_row_pos_1 = start_pos
            return x_col_pos_1, y_row_pos_1

        # Case when car finishes
        elif pos in track.finish_cells:
            x_col_pos_1, y_row_pos_1 = pos
            return x_col_pos_1, y_row_pos_1

    # Otherwise, return intended position
    return x_col_pos_1, y_row_pos_1


def compute_temp(temp: float, dissipation_frac: float = 0.999) -> float:
    """
    Compute temperature decrease resulting from heat loss.
    :param temp: Temperature at time t
    :param dissipation_frac: Temperature decrease over one time step
    :return: Temperature at time t + 1
    """
    return temp * dissipation_frac


def epsilon_greedy_policy(pos_0: tuple, vel_0: tuple, state_actions: pd.DataFrame, epsilon: float = EPSILON) -> tuple:
    """
    Choose policy.
    :param pos_0: x-y position pair
    :param vel_0: x-y velocity pair
    :param state_actions: State-actions table
    :return: x-y acceleration pair
    """
    x_col_pos_0, y_row_pos_0 = pos_0
    x_col_vel_0, y_row_vel_0 = vel_0

    actions = state_actions.loc[x_col_pos_0].loc[y_row_pos_0].loc[x_col_vel_0].loc[y_row_vel_0].reset_index()
    exploit_action = actions.iloc[0].to_dict()
    explore_action = actions.sample(n=1).iloc[0].to_dict()
    # Select action: Explore or exploit
    action = exploit_action if np.random.random() > epsilon else explore_action
    action.update(dict(x_col_vel=x_col_vel_0, y_row_vel=y_row_vel_0))
    return action["x_col_acc"], action["y_row_acc"]


def is_terminal(q: float) -> bool:
    """
    Determine if Q corresponds to terminal state.
    :param q: Q value
    :return: True if Q is in a terminal state
    """
    return q == 0


def select_s_prime_index(pos_prime: tuple, vel_prime: tuple, states: pd.DataFrame) -> int:
    """
    Select index of position-velocity pairs.
    :param pos_prime: x-y position pair
    :param vel_prime: x-y velocity pair
    :param states: States table
    :return: Corresponding states table index
    """
    mask = (states["x_col_pos"] == pos_prime[0]) & (states["y_row_pos"] == pos_prime[1])
    mask &= (states["x_col_vel"] == vel_prime[0]) & (states["y_row_vel"] == vel_prime[1])
    return states[mask].index.values[0]


def softmax_policy(pos_0: tuple, vel_0: tuple, state_actions: pd.DataFrame, temp: float) -> tuple:
    """
    Choose policy.
    :param pos_0: x-y position pair
    :param vel_0: x-y velocity pair
    :param state_actions: State-actions table
    :param temp: Temperature
    :return: x-y acceleration pair
    """
    x_col_pos_0, y_row_pos_0 = pos_0
    x_col_vel_0, y_row_vel_0 = vel_0

    actions = state_actions.loc[x_col_pos_0].loc[y_row_pos_0].loc[x_col_vel_0].loc[y_row_vel_0].reset_index()
    numerator = np.exp(actions["q"].abs() / temp)
    denominator = numerator.sum()
    actions["cum_probs"] = (numerator / denominator).cumsum()

    # Randomly select action
    mask = actions["cum_probs"] > np.random.random()
    action = actions[mask][["x_col_acc", "y_row_acc"]].iloc[0].to_dict()
    return action["x_col_acc"], action["y_row_acc"]


def state_action_dict(pos: tuple, vel: tuple, acc: tuple, state_actions: pd.DataFrame) -> dict:
    """
    Retrieve state-action dictionary from state-action dataframe.
    :param pos: x-y position pair
    :param vel: x-y velocity pair
    :param acc: x-y acceleration pair
    :param state_actions: Table of state-action pairs
    :return: State-action dictionary
    """
    return state_actions.loc[pos[0]].loc[pos[1]].loc[vel[0]].loc[vel[1]].loc[acc[0]].loc[acc[1]].to_dict()


def update_state_actions(new_q, pos: tuple, vel: tuple, acc: tuple, state_actions: pd.DataFrame) -> pd.DataFrame:
    """
    Update Q value of state actions table.
    :param new_q: New Q value
    :param pos: x-y position pair
    :param vel: x-y velocity pair
    :param acc: x-y acceleration pair
    :param state_actions: State-actions table
    :return: Updated state-actions table
    """
    row = state_actions.loc[pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]]
    row.loc["q"] = new_q
    row.loc["t"] = row.loc["t"] + 1
    state_actions.loc[pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]] = row.values
    return state_actions
