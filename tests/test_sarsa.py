#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, test_sarsa.py

"""
# Standard library imports
from pathlib import Path
import warnings

# Third party imports
import pandas as pd

# Local imports
from p5.settings import *
from p5.q_learning_sarsa import epsilon_greedy_policy, compute_position, compute_temp, is_terminal, \
    select_s_prime_index, softmax_policy, state_action_dict, update_state_actions
from p5.track import Track
from p5.utils import compute_velocity, realize_action

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
# Select policy
policy = softmax_policy


def test_sarsa():
    """
    Test value iteration algorithm.
    """
    # Iterate over each dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for track_src in track_srcs:
        for oob_penalty in OOB_PENALTIES:
            print(track_src.stem)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Make the track and possible states
            track = Track(track_src)
            track.prep_track()
            track.make_states()
            track.make_state_actions()
            track.sort_state_actions()
            state_actions = track.state_actions.copy()
            ix = track.states.index.values[0]
            # TODO: Select nearest non-finish, non-wall square?
            # TODO: Global temp or temp that is function of number of visits to current state?
            temp = INIT_TEMP

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Train over episodes
            ix_check = 5 * [None]
            history = []
            for episode in range(20000):
                state_actions.sort_values(by="q", ascending=False, inplace=True)

                # Choose action using policy derived from Q
                base_state_di = track.states.loc[ix].to_dict()
                pos = base_state_di["x_col_pos"], base_state_di["y_row_pos"]
                vel = base_state_di["x_col_vel"], base_state_di["y_row_vel"]

                # Apply policy to get action
                if policy == softmax_policy:
                    acc = softmax_policy(pos, vel, state_actions, temp)
                else:
                    acc = epsilon_greedy_policy(pos, vel, state_actions, epsilon=EPSILON)
                state_action_di = state_action_dict(pos, vel, acc, state_actions)
                q = state_action_di["q"]
                r = state_action_di["r"]

                # Find Q of s prime-a prime pair
                acc_real = realize_action(acc)
                vel_prime = compute_velocity(vel, acc_real)
                pos_prime = compute_position(pos, vel_prime, track)
                acc_prime = epsilon_greedy_policy(pos_prime, vel_prime, state_actions)
                state_action_prime_di = state_action_dict(pos_prime, vel_prime, acc_prime, state_actions)
                q_prime = state_action_prime_di["q"]

                # Compute new Q
                new_q = q + ETA * (r + GAMMA * q_prime - q)  # Don't change ETA
                assert new_q <= 0
                state_actions = update_state_actions(new_q, pos, vel, acc, state_actions)

                sanity_check = state_actions.loc[pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]].loc["q"]
                print(f"Episode {episode}, index {ix}, temp={temp:.3f}, q={sanity_check:.3f}")

                # Update index corresponding to s' and update temperature
                ix_check.append(ix)
                ix_check = ix_check[-5:]
                temp = compute_temp(temp, dissipation_frac=TEMP_DISSIPATION_FRAC)
                # Start somewhere else if Q corresponds to terminal state or if algo stuck in same state-action pair
                history.append(state_actions.loc[pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]].to_frame().transpose())

                if is_terminal(new_q) or (len(set(ix_check)) == 1):
                    print("\tReset next state")
                    track.sort_state_actions()
                    ix = track.states.index.values[0]
                # If one of last 4 episodes have nonzero Q values, take s' as next state
                else:
                    ix = select_s_prime_index(pos_prime, vel_prime, track.states)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Organize and save output
            history = pd.concat(history)
            names = ["x_col_pos", "y_row_pos", "x_col_vel", "y_row_vel", "x_col_acc", "y_row_acc"]
            mix = pd.MultiIndex.from_tuples(history.index.values, names=names)
            history.index = mix

            state_actions_dst = OUT_DIR / f"sarsa_state_actions_{track_src.stem}_{oob_penalty}.csv"
            history_dst = OUT_DIR / f"sarsa_history_{track_src.stem}_{oob_penalty}.csv"
            state_actions.to_csv(state_actions_dst)
            history.to_csv(history_dst)


if __name__ == "__main__":
    test_sarsa()
