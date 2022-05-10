#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, track.py

This module provides the Track class, which builds out the track, all states, and state-action pairs.

"""
# Standard library imports
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from p5.settings import ACTIONS, INIT_VAL
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
        self.track = pd.DataFrame([[x for x in y] for y in splits[1:]]).dropna(axis=0)

        self.start_cells = set()
        self.go_cells = set()
        self.wall_cells = set()
        self.finish_cells = set()
        self.states = None
        self.state_actions = None
        self.positions = None
        self.order = None

    def __repr__(self):
        return self.track.head().to_string()

    def find_nearest(self, dst_cells: list) -> list:
        """
        Find the nearest cell to the given set of cells.
        :param dst_cells: Cells to find other nearest cells to
        :return: Nearest cells
        """
        nearest = []
        temp = self.positions.copy()
        for dst_cell in dst_cells:
            pos_arr = temp.copy()[["x_col_pos", "y_row_pos"]].values
            temp["temp"] = minkowski_distance(np.array(dst_cell), pos_arr, p=1)
            mask = temp["order"] == np.inf
            if mask.sum() > 0:
                series = temp[mask].sort_values(by="temp").iloc[0]
                nearest.append((series.loc["x_col_pos"], series.loc["y_row_pos"]))
        return nearest

    def get_reward(self, pos: tuple) -> int:
        """
        Look up cell reward.
        :param pos: Column-row tuple
        :return: Corresponding reward
        """
        return 0 if pos in self.finish_cells else -1

    def make_order(self):
        """
        Make order table, which we use to prioritize cells to visit.
        """

        cols = ["space", "y_row_pos", "x_col_pos"]
        self.positions = self.states.copy()[cols].drop_duplicates()
        self.positions["order"] = np.inf
        mask = self.positions["space"] == "F"
        self.positions.loc[mask, "order"] = 0
        dst_cells = list(self.finish_cells)
        order = 1
        while True:
            nearest = self.find_nearest(dst_cells)
            if nearest:
                self.update_order(nearest, order)
                dst_cells = nearest
                order += 1
            else:
                break
        assert self.positions.order.max() < np.inf
        self.positions.order = self.positions.order.astype(int)
        cols = ["x_col_pos", "y_row_pos"]
        self.positions = self.positions.sort_values(by=cols).set_index(cols)
        self.order = self.track.copy()
        # Make order table as a way to sanity check assignments
        for index, row in self.positions.iterrows():
            x_col_pos, y_row_pos = index
            self.order.loc[y_row_pos].loc[x_col_pos] = row.order
        on = ["x_col_pos", "y_row_pos"]
        self.states = self.states.merge(self.positions.reset_index().drop(axis=1, labels="space"), on=on, how="left")

    def make_state_actions(self, velocities: list, init_val: int = -1) -> pd.DataFrame:
        """
        Make state-action pairs.
        :param velocities: List of allowable velocities
        :param init_val: Initial Q value
        :return: State-actions table
        For use with Q learning and SARSA.
        """
        state_actions = []
        for y_row_pos, row in self.track.iterrows():
            for x_col_pos, space in row.to_dict().items():
                if space != "#":
                    for y_row_vel in velocities:
                        for x_col_vel in velocities:
                            for action in ACTIONS:
                                x_col_acc, y_row_acc = action
                                r = 0 if space == "F" else -1
                                q = 0 if space == "F" else init_val
                                fin = True if space == "F" else False
                                state_action = dict(space=space, y_row_pos=y_row_pos, x_col_pos=x_col_pos,
                                                    y_row_vel=y_row_vel, x_col_vel=x_col_vel, x_col_acc=x_col_acc,
                                                    y_row_acc=y_row_acc, r=r, fin=fin, q=q, t=0, ep=-1)
                                state_actions.append(state_action)
        state_actions = pd.DataFrame(state_actions)
        index = ["x_col_pos", "y_row_pos", "x_col_vel", "y_row_vel", "x_col_acc", "y_row_acc"]
        self.state_actions = state_actions.set_index(index)
        return self.state_actions

    def make_states(self, velocities: list) -> pd.DataFrame:
        """
        Make states for value iteration algorithm.
        :param velocities: List of allowable velocities
        :return: States table
        """
        states = []
        for y_row_pos, row in self.track.iterrows():
            for x_col_pos, space in row.to_dict().items():
                if space != "#":
                    for y_row_vel in velocities:
                        for x_col_vel in velocities:
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

    def sort_states(self) -> pd.DataFrame:
        """
        Sort state-action pairs on the basis of nearness to finish squares.
        :return: Updated state-actions table
        """
        self.states = self.states.sample(frac=1)
        self.states.sort_values(by=["fin", "t", "order"], ascending=[True, True, True], inplace=True)
        return self.states

    def update_order(self, nearest: list, order: int):
        """
        Update the order column of the positions table using the nearest points to the target point.
        :param nearest: Nearest x-y position
        :param order: Order of nearest x-y position
        """
        for x_col_pos, y_row_pos in nearest:
            mask = (self.positions["x_col_pos"] == x_col_pos) & (self.positions["y_row_pos"] == y_row_pos)
            self.positions.loc[mask, "order"] = order


def mean_finish_position(states: pd.DataFrame) -> dict:
    """
    Compute mean finish position.
    :param states: States table
    :return: Mean x-y finish position
    """
    mask = states.space == "F"
    return states[mask][["x_col_pos", "y_row_pos"]].drop_duplicates().mean().to_dict()
