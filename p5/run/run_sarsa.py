#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, run_sarsa.py
Test SARSA algorithm.
"""
# Standard library imports
from pathlib import Path
import typing as t
import warnings

# Local imports
from p5.settings import EPSILON, ETA, GAMMA, INIT_TEMP, MIN_TEMP, OOB_PENALTIES, TEMP_DISSIPATION_FRAC, VELOCITIES
from p5.q_learning_sarsa import compute_position, compute_temp, epsilon_greedy_policy, is_terminal, save_history, \
    softmax_policy, state_action_dict, update_state_actions, update_states
from p5.track import Track
from p5.utils import compute_velocity, realize_action

warnings.filterwarnings('ignore')


def run_sarsa(src_dir: Path, dst_dir: Path, policy: t.Callable = softmax_policy):
    """
    Run SARSA algorithm for provided input track files.
    :param src_dir: Path to source / input directory containing track files
    :param dst_dir: Path to destination / output directory
    :param policy: Softmax or greedy epsilon
    """
    track_srcs = [x for x in src_dir.iterdir() if x.stem == "R-track"]
    # Iterate over each dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for track_src in track_srcs:
        for oob_penalty in OOB_PENALTIES:
            print(track_src.stem)

            temp = INIT_TEMP
            ix_check = 5 * [None]
            history = []

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Make the track and possible states
            track = Track(track_src, oob_penalty=oob_penalty)
            track.prep_track()
            track.make_states(velocities=VELOCITIES)
            track.make_state_actions(velocities=VELOCITIES)
            track.make_order()
            track.sort_states()

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize state
            ix = track.states.index.values[0]
            base_state_di = track.states.loc[ix].to_dict()
            pos = base_state_di["x_col_pos"], base_state_di["y_row_pos"]
            vel = base_state_di["x_col_vel"], base_state_di["y_row_vel"]

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Choose action using policy
            if policy == softmax_policy:
                acc = softmax_policy(pos, vel, track.state_actions, temp)
            else:
                acc = epsilon_greedy_policy(pos, vel, track.state_actions, epsilon=EPSILON)

            # Add first state-action pair to history
            history.append(
                track.state_actions.loc[pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]].to_frame().transpose())

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # For all episodes
            n_unvisited = len(track.state_actions)
            frac_unvisited = 1
            episode = 0
            ct = 0
            while frac_unvisited > 0:
                episode += 1
                print(f"Episode {episode}")

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize state
                track.sort_states()
                ix = track.states.index.values[0]
                base_state_di = track.states.loc[ix].to_dict()
                pos = base_state_di["x_col_pos"], base_state_di["y_row_pos"]
                vel = base_state_di["x_col_vel"], base_state_di["y_row_vel"]

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Choose action using policy
                if policy == softmax_policy:
                    acc = softmax_policy(pos, vel, track.state_actions, temp)
                else:
                    acc = epsilon_greedy_policy(pos, vel, track.state_actions, epsilon=EPSILON)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Repeat
                while not is_terminal(pos, vel, track.state_actions):
                    ct += 1

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Take action a and observe r and s'
                    state_action_di = state_action_dict(pos, vel, acc, track.state_actions)
                    q = state_action_di["q"]
                    acc_real = realize_action(acc)
                    vel_prime = compute_velocity(vel, acc_real)
                    pos_prime = compute_position(pos, vel_prime, track)
                    r = track.get_reward(pos_prime)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Choose a' using policy
                    if policy == softmax_policy:
                        acc_prime = softmax_policy(pos_prime, vel_prime, track.state_actions, temp)
                    else:
                        acc_prime = epsilon_greedy_policy(pos_prime, vel_prime, track.state_actions, epsilon=EPSILON)
                    state_action_prime_di = state_action_dict(pos_prime, vel_prime, acc_prime, track.state_actions)
                    q_prime = state_action_prime_di["q"]

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update Q(s, a)
                    new_q = q + ETA * (r + GAMMA * q_prime - q)
                    assert new_q <= 0
                    track.state_actions = update_state_actions(new_q, pos, vel, acc, episode, track.state_actions)
                    track.states = update_states(track.states, track.state_actions)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update temperature and append to history
                    temp = compute_temp(temp, dissipation_frac=TEMP_DISSIPATION_FRAC, min_temp=MIN_TEMP)
                    history.append(
                        track.state_actions.loc[pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]].to_frame().transpose())

                    # Print status
                    t = track.state_actions.loc[pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]].loc["t"]
                    print(
                        f"\tct={ct}, t={t}, frac_un={frac_unvisited:.4f}, n_un={n_unvisited}, pos={pos}, vel={vel}, acc={acc}, temp={temp:.1f}, q={q:.2f}, new_q={new_q:.2f}")

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update state, action, and fraction of state-action pairs
                    acc = acc_prime
                    vel = vel_prime
                    pos = pos_prime

                    n_unvisited = (track.state_actions.t == 0).sum()
                    frac_unvisited = n_unvisited / len(track.state_actions)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Save intermediate outputs
                    if ct % 50000 == 0:
                        state_actions_dst = dst_dir / f"sarsa_state_actions_{track_src.stem}_{oob_penalty}_{ct}.csv"
                        history_dst = dst_dir / f"sarsa_history_{track_src.stem}_{oob_penalty}_{ct}.csv"
                        track.state_actions.to_csv(state_actions_dst)
                        save_history(history, history_dst)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Organize and save final output
            state_actions_dst = dst_dir / f"sarsa_state_actions_{track_src.stem}_{oob_penalty}.csv"
            history_dst = dst_dir / f"sarsa_history_{track_src.stem}_{oob_penalty}.csv"
            track.state_actions.to_csv(state_actions_dst)
            save_history(history, history_dst)
