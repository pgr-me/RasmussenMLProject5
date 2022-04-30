#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, utils.py

This module provides miscellaneous utility functions.

"""
# Third party libraries
import pandas as pd


def bresenham(x_col_0: int, y_row_0: int, x_col_1: int, y_row_1: int):
    """
    Compute the cells visited between two raster cells using the Breseham algorithm.
    :param x_col_0: Starting x
    :param y_row_0: Starting y
    :param x_col_1: Ending x
    :param y_row_1: Ending y
    First element of returned array is the starting point.
    Source: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm#:~:text=Bresenham%27s%20line%20algorithm%20is%20a%20line%20drawing%20algorithm,approximation%20to%20a%20straight%20line%20between%20two%20points.
    TinyURL: https://tinyurl.com/2p9692h6
    """
    dx = abs(x_col_1 - x_col_0)
    sx = 1 if x_col_0 < x_col_1 else -1
    dy = -abs(y_row_1 - y_row_0)
    sy = 1 if y_row_0 < y_row_1 else -1
    error = dx + dy
    output = []
    while True:
        output.append((x_col_0, y_row_0))
        if (x_col_0 == x_col_1) & (y_row_0 == y_row_1):
            return output
        e2 = 2 * error
        if e2 >= dy:
            if x_col_0 == x_col_1:
                return output
            error += dy
            x_col_0 += sx
        if e2 <= dx:
            if y_row_0 == y_row_1:
                return output
            error += dx
            y_row_0 += sy


def in_bounds(pos: tuple, track: pd.DataFrame):
    row, col = pos
    if (row >= track.shape[0]) or (row < 0):
        return False
    if (col >= track.shape[1]) or (col < 0):
        return False
    return True


def is_finished(pos, track: pd.DataFrame):
    if track.iloc[pos[0], pos[1]] == "F":
        return True
    return False
