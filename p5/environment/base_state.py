#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, utils.py

This module provides the base state class from which other classes inherit.

"""
# Third party imports
import numpy as np
import pandas as pd

# Local imports
from p5.environment.track import Track
from p5.settings import VELOCITIES


class BaseState:
    def __init__(self,
                 x_col_pos: int, y_row_pos: int,
                 x_col_vel: int, y_row_vel: int,
                 x_col_init_acc: int, y_row_init_acc: int,
                 t: int,
                 track: Track,
                 oob_penalty: str = "stay-in-place",
                 success_rate: float = 0.2):
        self.x_col_pos_0 = x_col_pos
        self.y_row_pos_0 = y_row_pos
        self.x_col_vel_0 = x_col_vel
        self.y_row_vel_0 = y_row_vel
        self.x_col_succeed_acc = x_col_init_acc
        self.y_row_succeed_acc = y_row_init_acc
        self.t_0 = t

        self.track = track
        self.n_start_cells = len(self.track.start_cells)
        self.oob_penalty = oob_penalty
        self.success_rate = success_rate

        self.x_col_pos_1 = None
        self.y_row_pos_1 = None
        self.x_col_vel_1 = None
        self.y_row_vel_1 = None
        self.x_col_acc = None
        self.y_row_acc = None
        self.t_1 = None
        self.oob = None
        self.fin = None
        self.states = None

    def __repr__(self):
        xpos0, ypos0 = self.x_col_pos_0, self.y_row_pos_0
        xvel0, yvel0 = self.x_col_vel_0, self.y_row_vel_0
        xacc0, yacc0 = self.x_col_succeed_acc, self.y_row_succeed_acc

        xpos1, ypos1 = self.x_col_pos_1, self.y_row_pos_1
        xvel1, yvel1 = self.x_col_vel_1, self.y_row_vel_1
        xacc1, yacc1 = self.x_col_acc, self.y_row_acc

        str_ = f"Pos={xpos0, ypos0}, Vel={xvel0, yvel0}, Acc={xacc0, yacc0} @ t={self.t_0}; "
        str_ += f"Pos={xpos1, ypos1}, Vel={xvel1, yvel1}, Acc={xacc1, yacc1} @ t={self.t_1}"
        return str_

    def reset(self):
        self.x_col_pos_1 = None
        self.y_row_pos_1 = None
        self.x_col_vel_1 = None
        self.y_row_vel_1 = None
        self.x_col_acc = None
        self.y_row_acc = None
        self.t_1 = None
        self.oob = None
        self.fin = None
