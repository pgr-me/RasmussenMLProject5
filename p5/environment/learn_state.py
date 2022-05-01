#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, utils.py

This module provides the DriverState class, a state-level component of the RL training. The agent learns the policy in
this module.

"""
# Third party imports
import pandas as pd

# Local imports
from p5.environment.track import Track
from p5.utils import bresenham
from p5.settings import ACTIONS, GAMMA, VEL_MAX, VEL_MIN, P_ACC_SUCCEED, P_ACC_FAIL


def learn_state(track: Track, states, ix, oob_penalty) -> dict:
    """
    Learn one state.
    :param track: Track object
    :param states: Dataframe of states
    :param ix: States dataframe index
    :param oob_penalty: Out-of-bounds penalty
    :return: Best value and action
    """
    # Extract a dictionary of state attributes: space, position, velocity, reward, time, etc.
    state_di = states.loc[ix].to_dict()

    best_action, best_Q_sa = None, -float("inf")
    for action in ACTIONS:
        # Case when action succeeds
        x_col_acc, y_row_acc = action
        succeed_vel = update_velocity(state_di, x_col_acc, y_row_acc)
        succeed_pos_li: list = update_position(state_di, track, succeed_vel[0], succeed_vel[1], oob_penalty=oob_penalty,
                                               succeed=True)

        # Case when action fails
        fail_vel = update_velocity(state_di, 0, 0)
        fail_pos_li: list = update_position(state_di, track, fail_vel[0], fail_vel[1], oob_penalty=oob_penalty,
                                            succeed=False)

        # Combine success and failure cases
        actions = pd.concat([pd.DataFrame(succeed_pos_li), pd.DataFrame(fail_pos_li)])
        actions = compute_state_weights(actions)
        di = {"x_col_vel_1": "x_col_vel", "y_row_vel_1": "y_row_vel", "x_col_pos_1": "x_col_pos",
              "y_row_pos_1": "y_row_pos"}
        labels = ["x_col_vel", "y_row_vel", "x_col_pos", "y_row_pos", "val"]
        on = [x for x in labels if x != "val"]
        actions = actions.rename(columns=di).merge(states[labels], on=on, how="left")

        # Compute the expected value

        Q_sa: float = state_di["r"] + GAMMA * (actions.wt * actions.val).sum()  # * state_di["prev_val"]
        if Q_sa > best_Q_sa:
            best_Q_sa = Q_sa
            best_action = action
    best_dict = dict(val=best_Q_sa, best_x_col_a=best_action[0], best_y_row_a=best_action[1])
    return best_dict


def update_position(state_di: dict, track: Track, x_col_vel_1, y_row_vel_1, oob_penalty="stay-in-place",
                    succeed=True) -> list:
    """
    Update position based on velocity.
    """
    x_col_pos_0, y_row_pos_0 = state_di["x_col_pos"], state_di["y_row_pos"]
    x_col_pos_1 = x_col_pos_0 + x_col_vel_1
    y_row_pos_1 = y_row_pos_0 + y_row_vel_1
    reward = track.get_reward((x_col_pos_1, y_row_pos_1))

    positions = []
    for pos in bresenham(x_col_pos_0, y_row_pos_0, x_col_pos_1, y_row_pos_1):

        # Case when driver hits wall
        if pos in track.wall_cells:

            # Case when OOB (out-of-bounds) penalty is to remain in current position
            if oob_penalty == "stay-in-place":
                x_col_pos_1 = x_col_pos_0
                y_row_pos_1 = y_row_pos_0
                reward = track.get_reward((x_col_pos_1, y_row_pos_1))
                di = dict(x_col_vel_1=x_col_vel_1, y_row_vel_1=y_row_vel_1, x_col_pos_1=x_col_pos_1,
                          y_row_pos_1=y_row_pos_1, ct=1, fin=False, r=reward, s=succeed)
                positions.append(di)

            # Case when OOB penalty is to return to one of the start positions
            else:
                prob = 1 / len(track.start_cells)
                for start_pos in track.start_cells:
                    reward = track.get_reward(start_pos)
                    di = dict(x_col_vel_1=x_col_vel_1, y_row_vel_1=y_row_vel_1, x_col_pos_1=start_pos[0],
                              y_row_pos_1=start_pos[1], ct=prob, fin=False, r=reward,
                              s=succeed)
                    positions.append(di)

            return positions

        # Case when driver reaches finish line
        elif pos in track.finish_cells:
            x_col_pos_1, y_row_pos_1 = pos
            reward = track.get_reward(pos)
            di = dict(x_col_vel_1=x_col_vel_1, y_row_vel_1=y_row_vel_1, x_col_pos_1=x_col_pos_1,
                      y_row_pos_1=y_row_pos_1, ct=1, fin=True, r=reward, s=succeed)
            positions.append(di)
            # Stop if driver reaches finish line
            return positions

    # Case when driver reaches a non-wall destination
    fin = True if (x_col_pos_1, y_row_pos_1) in track.finish_cells else False
    positions.append(
        dict(x_col_vel_1=x_col_vel_1, y_row_vel_1=y_row_vel_1, x_col_pos_1=x_col_pos_1, y_row_pos_1=y_row_pos_1, ct=1,
             fin=fin, r=reward, s=succeed))
    return positions


def update_velocity(state_di: dict, x_col_acc, y_row_acc):
    """
    Update car speed based on acceleration.
    """
    x_col_vel_1 = state_di["x_col_vel"] + x_col_acc
    y_row_vel_1 = state_di["y_row_vel"] + y_row_acc

    # Update velocities subject to max and min velocity limits
    x_col_vel_1 = VEL_MAX if x_col_vel_1 > VEL_MAX else x_col_vel_1
    x_col_vel_1 = VEL_MIN if x_col_vel_1 < VEL_MIN else x_col_vel_1
    y_row_vel_1 = VEL_MAX if y_row_vel_1 > VEL_MAX else y_row_vel_1
    y_row_vel_1 = VEL_MIN if y_row_vel_1 < VEL_MIN else y_row_vel_1

    return x_col_vel_1, y_row_vel_1


def compute_state_weights(action_df: pd.DataFrame) -> pd.DataFrame:
    mask = action_df["s"]
    succeed_tot = mask.sum()
    fail_tot = len(action_df) - succeed_tot
    action_df.loc[mask, "wt"] = P_ACC_SUCCEED * action_df.loc[mask, "ct"] / succeed_tot
    action_df.loc[~mask, "wt"] = P_ACC_FAIL * action_df.loc[mask, "ct"] / fail_tot
    return action_df
