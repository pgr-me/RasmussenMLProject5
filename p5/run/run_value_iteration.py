#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, run_value_iteration.py

"""
# Standard library imports
from pathlib import Path
import multiprocessing as mp
import warnings

# Third party imports
import pandas as pd

# Local imports
from p5.settings import N_CORES, OOB_PENALTIES, VALUE_ITERATION_TERMINATION_THRESHOLD, VELOCITIES
from p5.value_iteration import learn_state
from p5.track import Track

warnings.filterwarnings('ignore')


def run_value_iteration(src_dir: Path, dst_dir: Path):
    """
    Test value iteration algorithm.
    :param src_dir: Path to source / input directory containing track files
    :param dst_dir: Path to destination / output directory
    """
    track_srcs = [x for x in src_dir.iterdir() if x.stem == "toy-track"]
    # Iterate over each dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for track_src in track_srcs:
        for oob_penalty in OOB_PENALTIES:
            print(f"{track_src.stem}, OOB penalty={oob_penalty}")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Make the track and possible states
            track = Track(track_src)
            track.prep_track()
            track.make_states(velocities=VELOCITIES)
            states = track.states.copy()
            indices = track.states.index.values
            learning_curve = []

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Train over episodes
            for episode in range(100):
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

                if episode % 10 == 0:
                    # Save intermediate outputs
                    states_dst = dst_dir / f"value_iter_states_{track_src.stem}_{oob_penalty}_{episode}.csv"
                    learning_curve_dst = dst_dir / f"value_iter_learning_curve_{track_src.stem}_{oob_penalty}_{episode}.csv"
                    states.to_csv(states_dst, index=False)
                    pd.DataFrame(learning_curve).to_csv(learning_curve_dst, index=False)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save output
            states_dst = dst_dir / f"value_iter_states_{track_src.stem}_{oob_penalty}.csv"
            learning_curve_dst = dst_dir / f"value_iter_learning_curve_{track_src.stem}_{oob_penalty}.csv"
            states.to_csv(states_dst, index=False)
            pd.DataFrame(learning_curve).to_csv(learning_curve_dst, index=False)
