#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, test_value_iteration.py

"""
# Standard library imports
from pathlib import Path
import warnings

# Local imports
from p5.settings import *
from p5.q_learning import epsilon_greedy_policy, compute_position, compute_temp, state_action_dict, \
    select_s_prime_index, softmax_policy, update_state_actions
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


def test_value_iteration():
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
            states = track.states.copy()
            indices = track.states.index.values
            # TODO: Select nearest non-finish, non-wall square?
            ix = indices[0]
            # TODO: Global temp or temp that is function of number of visits to current state?
            temp = INIT_TEMP

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Train over episodes
            for episode in range(10000):
                state_actions.sort_values(by="q", ascending=False, inplace=True)

                # TODO: I iterate over states? Not state-action pairs, correct?
                # Choose action using policy derived from Q
                base_state_di = track.states.loc[ix].to_dict()
                pos = base_state_di["x_col_pos"], base_state_di["y_row_pos"]
                vel = base_state_di["x_col_vel"], base_state_di["y_row_vel"]

                # Case when car is already on finish line: this state is a sink and we need not go further
                # TODO: Is it a good idea to have the finish states act as "sinks" to basically stop further computation?
                # TODO: Wouldn't Q' just be zero in this case?
                # if (x_col_pos, y_row_pos) in track.finish_cells:
                #    x_col_acc, y_row_acc = 0, 0
                #    x_col_vel_1, y_row_vel_1 = 0, 0
                #    x_col_pos_1, y_row_pos_1 = x_col_pos_0, y_row_pos_0

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
                # TODO: I reduce ETA over time, right?
                new_q = q + ETA * (r + GAMMA * q_prime - q)
                state_actions = update_state_actions(new_q, pos, vel, acc, state_actions)

                sanity_check = state_actions.loc[pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]].loc["q"]
                print(f"Episode {episode}, index {ix}, q={sanity_check}, temp={temp:.3f}")

                # TODO: What should I save for the learning curve?
                # Update index corresponding to s' and update temperature
                ix = select_s_prime_index(pos_prime, vel_prime, states)
                temp = compute_temp(temp, dissipation_frac=TEMP_DISSIPATION_FRAC)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save output
            dst = OUT_DIR / f"q_learning_{track_src.stem}_{oob_penalty}.csv"
            state_actions.to_csv(dst, index=False)


if __name__ == "__main__":
    test_value_iteration()
