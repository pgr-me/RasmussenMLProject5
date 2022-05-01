#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, test_value_iteration.py

"""
# Standard library imports
from pathlib import Path
import multiprocessing as mp
import warnings

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from p5.settings import *
from p5.environment.learn_state import compute_state_weights, update_position, update_velocity
from p5.environment.track import Track

warnings.filterwarnings('ignore')

# Define constants
TEST_DIR = Path(".").absolute()
REPO_DIR = TEST_DIR.parent
P4_DIR = REPO_DIR / "p5"
DATA_DIR = REPO_DIR / "data"
IN_DIR = DATA_DIR / "in"
OUT_DIR = DATA_DIR / "out"
THRESH = 0.01
K_FOLDS = 5
VAL_FRAC = 0.2

track_srcs = [x for x in IN_DIR.iterdir() if x.stem == "toy-track"]
OOB_PENALTIES = ["stay-in-place", "back-to-beginning"]


def learn_state(track, states, ix, oob_penalty):
    # series = states.loc[ix]
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
        labels = ["x_col_vel", "y_row_vel", "x_col_pos", "y_row_pos", "prev_val"]
        on = [x for x in labels if x != "prev_val"]
        actions = actions.rename(columns=di).merge(states[labels], on=on, how="left")

        # Compute the expected value

        Q_sa: float = state_di["r"] + GAMMA * (actions.wt * actions.prev_val).sum()  # * state_di["prev_val"]
        if Q_sa > best_Q_sa:
            best_Q_sa = Q_sa
            best_action = action
    best_dict = dict(val=best_Q_sa, best_x_col_a=best_action[0], best_y_row_a=best_action[1])
    return best_dict


def test_value_iteration():
    # Iterate over each dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for oob_penalty in OOB_PENALTIES:
        for track_src in track_srcs:
            print(track_src.stem)

            # Make the track and possible states
            track = Track(track_src)
            track.prep_track()
            track.make_states()
            states = track.states.copy()
            indices = track.states.index.values
            for i in range(10):
                print(i)
                states["t"] = states["t"] + 1
                with mp.Pool(processes=N_CORES) as pool:
                    best_li: list = pool.starmap(learn_state, [(track, states, ix, oob_penalty) for ix in indices])
                best_df = pd.DataFrame(best_li)
                states["prev_val"] = states["val"]
                labels = ["val", "best_x_col_a", "best_y_row_a"]

                states = states.drop(axis=1, labels=labels).join(best_df)

                ## Update track values and best action
                # states.loc[ix, "prev_val"] = states.loc[ix].loc["val"]
                # states.loc[ix, "val"] = best_Q_sa
                # states.loc[ix, "best_x_col_a"] = best_action[0]
                # states.loc[ix, "best_y_row_a"] = best_action[1]
                # pass

                # Exit loop if greatest improvement less than threshold
                max_diff = (states["val"] - states["prev_val"]).abs().max()
                # if max_diff <= VALUE_ITERATION_TERMINATION_THRESHOLD:
                #    break

                print('y')


if __name__ == "__main__":
    test_value_iteration()
