#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, test_q_learning.py

"""
# Standard library imports
from pathlib import Path
import warnings

# Third party imports
import pandas as pd

# Local imports
from p5.settings import *
from p5.q_learning_sarsa import epsilon_greedy_policy, compute_position, compute_temp, is_terminal, softmax_policy,\
    state_action_dict, update_state_actions
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

track_srcs = [x for x in IN_DIR.iterdir() if x.stem == "demo-track"]
OOB_PENALTIES = ["stay-in-place", "back-to-beginning"]
# Select policy
policy = softmax_policy


def test_q_learning():
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
            track.sort_states()
            # Choose action using policy derived from Q
            temp = INIT_TEMP

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Train over episodes
            history = []
            n_unvisited = (track.state_actions.t == 0).sum()
            episode = 0
            ct = 0
            while n_unvisited > 0:
                episode += 1
                print(f"Episode {episode}")

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize state
                track.sort_states()
                ix = track.states.index.values[0]
                base_state_di = track.states.loc[ix].to_dict()
                pos = base_state_di["x_col_pos"], base_state_di["y_row_pos"]
                vel = base_state_di["x_col_vel"], base_state_di["y_row_vel"]

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Repeat
                while not is_terminal(pos, vel, track.state_actions):
                    ct += 1

                    track.state_actions.sort_values(by="q", ascending=False, inplace=True)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Apply policy to get action
                    if policy == softmax_policy:
                        acc = softmax_policy(pos, vel, track.state_actions, temp)
                    else:
                        acc = epsilon_greedy_policy(pos, vel, track.state_actions, epsilon=EPSILON)
                    state_action_di = state_action_dict(pos, vel, acc, track.state_actions)
                    q = state_action_di["q"]
                    r = state_action_di["r"]

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Find Q of s prime-a prime pair
                    acc_real = realize_action(acc)
                    vel_prime = compute_velocity(vel, acc_real)
                    pos_prime = compute_position(pos, vel_prime, track)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Choose action that maximizes Q for state s'
                    acc_prime = track.state_actions.loc[pos_prime[0], pos_prime[1], vel_prime[0], vel_prime[1]] \
                        .sort_values(by="q", ascending=False).iloc[0].name
                    state_action_prime_di = state_action_dict(pos_prime, vel_prime, acc_prime, track.state_actions)
                    q_prime = state_action_prime_di["q"]

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute new Q
                    new_q = q + ETA * (r + GAMMA * q_prime - q)  # Keep ETA constant
                    assert new_q <= 0
                    track.state_actions = update_state_actions(new_q, pos, vel, acc, episode, track.state_actions)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update temperature and append to history
                    temp = compute_temp(temp, dissipation_frac=TEMP_DISSIPATION_FRAC, min_temp=MIN_TEMP)
                    frame = track.state_actions.loc[
                        pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]].to_frame().transpose()
                    history.append(frame)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Print status
                    t = track.state_actions.loc[pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]].loc["t"]
                    print(
                        f"\tct={ct}, t={t}, n_un={n_unvisited}, pos={pos}, vel={vel}, acc={acc}, temp={temp:.1f}, q={q:.2f}, new_q={new_q:.2f}")

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update state
                    pos, vel = pos_prime, vel_prime

                n_unvisited = (track.state_actions.t == 0).sum()

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Organize and save output
            history = pd.concat(history)
            names = ["x_col_pos", "y_row_pos", "x_col_vel", "y_row_vel", "x_col_acc", "y_row_acc"]
            mix = pd.MultiIndex.from_tuples(history.index.values, names=names)
            history.index = mix

            state_actions_dst = OUT_DIR / f"q_learning_state_actions_{track_src.stem}_{oob_penalty}.csv"
            history_dst = OUT_DIR / f"q_learning_history_{track_src.stem}_{oob_penalty}.csv"
            track.state_actions.to_csv(state_actions_dst)
            history.to_csv(history_dst)


if __name__ == "__main__":
    test_q_learning()
