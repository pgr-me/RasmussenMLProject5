#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, track.py

This module provides the Track class, which builds out the track and all states.

"""
# Standard library imports
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from p5.settings import ACTIONS, INIT_VAL, VELOCITIES
from p5.utils import minkowski_distance


class Track:
    def __init__(self, src: Path, oob_penalty: str = "stay-in-place"):
        """
        Construct the track object.
        :param src: Path to track text file
        :param oob_penalty: stay-in-place or back-to-beginning
        """
        self.src = src
        self.oob_penalty = oob_penalty
        with open(src) as f:
            self.text = f.read()
        splits = self.text.split("\n")
        self.track = pd.DataFrame([[x for x in y] for y in splits[1:]])

        self.start_cells = set()
        self.go_cells = set()
        self.wall_cells = set()
        self.finish_cells = set()
        self.states = None
        self.state_actions = None

    def __repr__(self):
        return self.track.head().to_string()

    def prep_track(self):
        """
        Prepare the track for use.
        """
        for y_row_ix, row in self.track.iterrows():
            for x_col_ix, val in row.iteritems():
                if val == "F":
                    self.finish_cells.add((x_col_ix, y_row_ix))
                elif val == "S":
                    self.start_cells.add((x_col_ix, y_row_ix))
                elif val == "#":
                    self.wall_cells.add((x_col_ix, y_row_ix))
                else:
                    self.go_cells.add((x_col_ix, y_row_ix))

    def get_reward(self, pos: tuple) -> int:
        """
        Look up cell reward.
        :param pos: Column-row tuple
        :return: Corresponding reward
        """
        return 0 if pos in self.finish_cells else -1

    def make_states(self) -> pd.DataFrame:
        """
        Make states for value iteration algorithm.
        :return: States table
        """
        states = []
        for y_row_pos, row in self.track.iterrows():
            for x_col_pos, space in row.to_dict().items():
                if space != "#":
                    for y_row_vel in VELOCITIES:
                        for x_col_vel in VELOCITIES:
                            r = 0 if space == "F" else -1
                            val = 0 if space == "F" else INIT_VAL
                            fin = True if space == "F" else False
                            state = dict(space=space, y_row_pos=y_row_pos, x_col_pos=x_col_pos, y_row_vel=y_row_vel,
                                         x_col_vel=x_col_vel, r=r, fin=fin, prev_val=0, val=val, t=0,
                                         best_x_col_a=np.nan, best_y_row_a=np.nan)
                            states.append(state)
        states = pd.DataFrame(states)
        self.states = states
        return self.states

    def make_state_actions(self) -> pd.DataFrame:
        """
        Make state-action pairs.
        :return: State-actions table
        For use with Q learning and SARSA.
        """
        state_actions = []
        for y_row_pos, row in self.track.iterrows():
            for x_col_pos, space in row.to_dict().items():
                if space != "#":
                    for y_row_vel in VELOCITIES:
                        for x_col_vel in VELOCITIES:
                            for action in ACTIONS:
                                x_col_acc, y_row_acc = action
                                r = 0 if space == "F" else -1
                                q = 0 if space == "F" else -10
                                fin = True if space == "F" else False
                                state_action = dict(space=space, y_row_pos=y_row_pos, x_col_pos=x_col_pos,
                                                    y_row_vel=y_row_vel, x_col_vel=x_col_vel, x_col_acc=x_col_acc,
                                                    y_row_acc=y_row_acc, r=r, fin=fin, q=0, t=0)
                                state_actions.append(state_action)
        state_actions = pd.DataFrame(state_actions)
        index = ["x_col_pos", "y_row_pos", "x_col_vel", "y_row_vel", "x_col_acc", "y_row_acc"]
        self.state_actions = state_actions.set_index(index)
        return self.state_actions

    def sort_state_actions(self) -> pd.DataFrame:
        """
        Sort state-action pairs on the basis of nearness to finish squares.
        :return: Updated state-actions table
        """
        mean_finish_pos = mean_finish_position(self.states)
        mean_finish_pos = np.array([mean_finish_pos["x_col_pos"], mean_finish_pos["y_row_pos"]])
        state_pos_arr = self.states.copy()[["x_col_pos", "y_row_pos"]].values
        self.states["fin_dist"] = minkowski_distance(mean_finish_pos, state_pos_arr)
        self.states.sort_values(by=["space", "t", "fin_dist"], ascending=[True, True, True], inplace=True)
        return self.states


def mean_finish_position(states: pd.DataFrame) -> dict:
    """
    Compute mean finish position.
    :param states: States table
    :return: Mean x-y finish position
    """
    mask = states.space == "F"
    return states[mask][["x_col_pos", "y_row_pos"]].drop_duplicates().mean().to_dict()
