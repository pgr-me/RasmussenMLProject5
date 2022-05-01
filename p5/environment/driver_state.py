#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 5, utils.py

This module provides the DriverState class, a state-level component of the simulation. The agent does not learn in this
module; it acts given the policy.


"""
# Third party imports
import numpy as np

# Local imports
from p5.environment.track import Track
from p5.environment.base_state import BaseState
from p5.utils import bresenham
from p5.settings import ACCELERATION, VEL_MAX, VEL_MIN


class DriverState(BaseState):
    def __init__(self, x_col_pos: int, y_row_pos: int, x_col_vel: int, y_row_vel: int, x_col_init_acc: int,
                 y_row_init_acc: int, t: int, track: Track, oob_penalty: str = "stay-in-place",
                 success_rate: float = 0.2):
        super().__init__(x_col_pos, y_row_pos, x_col_vel, y_row_vel, x_col_init_acc, y_row_init_acc, t, track,
                         oob_penalty, success_rate)

    def accelerate(self):
        """
        For use after model is trained.
        """
        if self.x_col_succeed_acc not in ACCELERATION:
            raise ValueError(f"x_col_init_acc is {self.x_col_succeed_acc} but must be one of {ACCELERATION}.")
        if self.y_row_succeed_acc not in ACCELERATION:
            raise ValueError(f"y_col_init_acc is {self.y_row_succeed_acc} but must be one of {ACCELERATION}.")

        if np.random.random() >= self.success_rate:
            self.x_col_acc, self.y_row_acc = self.x_col_succeed_acc, self.y_row_succeed_acc
        else:
            self.x_col_acc, self.y_row_acc = 0, 0

        return self.x_col_acc, self.y_row_acc

    def move(self):
        """
        For use after model is trained.
        """
        self.accelerate()
        self.update_velocity()
        self.update_position()
        self.t_1 = self.t_0 + 1

    def update_position(self):
        """
        Update position based on velocity.
        """
        x_col_pos_1 = self.x_col_pos_0 + self.x_col_vel_1
        y_row_pos_1 = self.y_row_pos_0 + self.y_row_vel_1

        path = bresenham(self.x_col_pos_0, self.y_row_pos_0, x_col_pos_1, y_row_pos_1)
        for pos in path:
            if pos in self.track.wall_cells:
                if self.oob_penalty == "stay-in-place":
                    x_col_pos_1 = self.x_col_pos_0
                    y_row_pos_1 = self.y_row_pos_0
                else:
                    start_pos_ix = np.random.choice(range(len(self.track.start_cells)))
                    start_pos = list(self.track.start_cells)[start_pos_ix]
                    x_col_pos_1, y_row_pos_1 = start_pos
                break
            elif pos in self.track.finish_cells:
                self.fin = True
                x_col_pos_1, y_row_pos_1 = pos
                break

        self.x_col_pos_1, self.y_row_pos_1 = x_col_pos_1, y_row_pos_1

        return self.x_col_pos_1, self.y_row_pos_1

    def update_velocity(self):
        """
        Update car speed based on acceleration.
        """
        x_col_vel = self.x_col_vel_0 + self.x_col_acc
        y_row_vel = self.y_row_vel_0 + self.y_row_acc

        # Update velocities subject to max and min velocity limits
        x_col_vel = VEL_MAX if x_col_vel > VEL_MAX else x_col_vel
        x_col_vel = VEL_MIN if x_col_vel < VEL_MIN else x_col_vel
        y_row_vel = VEL_MAX if y_row_vel > VEL_MAX else y_row_vel
        y_row_vel = VEL_MIN if y_row_vel < VEL_MIN else y_row_vel

        self.x_col_vel_1, self.y_row_vel_1 = x_col_vel, y_row_vel

        return (self.x_col_vel_1, self.y_row_vel_1)
