#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, test_value_iteration.py

"""
# Standard library imports
from pathlib import Path
import multiprocessing as mp
import warnings

# Third party imports
import pandas as pd

# Local imports
from p5.settings import *
from p5.value_iteration import learn_state
from p5.track import Track

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


def test_value_iteration():
    """
    Test value iteration algorithm.
    """
    # Iterate over each dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for track_src in track_srcs:
        for oob_penalty in OOB_PENALTIES:
            print(f"{track_src.stem}, OOB penalty={oob_penalty}")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Make the track and possible states
            track = Track(track_src)
            track.prep_track()
            track.make_states()
            states = track.states.copy()
            indices = track.states.index.values
            learning_curve = []

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Train over episodes
            for episode in range(20):
                states["t"] = states["t"] + 1

                # Learn the states
                with mp.Pool(processes=N_CORES) as pool:
                    best_li: list = pool.starmap(learn_state, [(track, states, ix, oob_penalty) for ix in indices])

                # Update the states dataframe
                best_df = pd.DataFrame(best_li)
                states["prev_val"] = states["val"]
                labels = ["val", "best_x_col_a", "best_y_row_a"]
                states = states.drop(axis=1, labels=labels).join(best_df)

                # Exit loop if greatest improvement less than threshold
                max_diff = (states["val"] - states["prev_val"]).abs().max()
                print(f'Iteration {episode}: max diff={max_diff:.2f}')
                learning_curve.append(dict(episode=episode, max_diff=max_diff))
                if max_diff <= VALUE_ITERATION_TERMINATION_THRESHOLD:
                    break

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save output
            states_dst = OUT_DIR / f"value_iteration_states_{track_src.stem}_{oob_penalty}.csv"
            learning_curve_dst = OUT_DIR / f"value_iteration_learning_curve_{track_src.stem}_{oob_penalty}.csv"
            states.to_csv(states_dst, index=False)
            pd.DataFrame(learning_curve).to_csv(learning_curve_dst, index=False)


if __name__ == "__main__":
    test_value_iteration()
