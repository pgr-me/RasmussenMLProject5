#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, utils.py

This module provides miscellaneous utility functions.

"""
# Third party imports
import numpy as np
import pandas as pd

# Local imports
from p5.settings import VELOCITIES


class Track:
    def __init__(self, src):
        with open(src) as f:
            self.text = f.read()
        splits = self.text.split("\n")
        self.track = pd.DataFrame([[x for x in y] for y in splits[1:]])

        self.start_cells = set()
        self.go_cells = set()
        self.wall_cells = set()
        self.finish_cells = set()
        self.states = None

    def __repr__(self):
        return self.track.head().to_string()

    def prep_track(self):
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
        Look up reward
        :param pos: Column-row tuple
        :return: Corresponding reward
        """
        return 0 if pos in self.finish_cells else -1

    def make_states(self) -> pd.DataFrame:
        """
        Make states.
        :return: States table
        """

        states = []
        for y_row_pos, row in self.track.iterrows():
            for x_col_pos, space in row.to_dict().items():
                if space != "#":
                    for y_row_vel in VELOCITIES:
                        for x_col_vel in VELOCITIES:
                            r = 0 if space == "F" else -1
                            fin = True if space == "F" else False
                            state = dict(space=space, y_row_pos=y_row_pos, x_col_pos=x_col_pos, y_row_vel=y_row_vel,
                                         x_col_vel=x_col_vel, r=r, fin=fin, prev_val=0, val=0, t=0, best_x_col_a=np.nan,
                                         best_y_row_a=np.nan)
                            states.append(state)
        states = pd.DataFrame(states)
        self.states = states
        return self.states
