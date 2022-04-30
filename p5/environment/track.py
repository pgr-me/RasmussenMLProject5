#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, utils.py

This module provides miscellaneous utility functions.

"""
import pandas as pd


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
