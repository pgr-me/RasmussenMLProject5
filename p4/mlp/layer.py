#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, layer.py

This module provides the Layer class for multi-layer perceptron training, validation, and prediction routines.

Sources: Lecture notes, Alpaydin, https://zerowithdot.com/mlp-backpropagation/, and
https://brilliant.org/wiki/backpropagation/
"""
# Standard library imports
import typing as t
# Third party libraries
import numpy as np
# Local imports
from p4.utils import bias, sigmoid


class Layer:
    def __init__(self, name: str, n_units: int, n_input_units: t.Union[int, None], apply_sigmoid: bool = True):
        self.name = name
        self.n_input_units = n_input_units
        self.n_units = n_units
        self.apply_sigmoid = apply_sigmoid
        self.W = None
        self.X = None
        self.Z = None

    def __repr__(self):
        return f"Layer {self.name}"

    def activation(self, Z):
        return sigmoid(Z) if self.apply_sigmoid else Z

    def activation_derivative(self, Z):
        return Z * (1 - Z) if self.apply_sigmoid else Z

    def backprop_error(self, error, Z):
        activation_derivative = self.activation_derivative(Z)
        delta_W = self.W.T.dot(error.T).T[:, 1:] * activation_derivative
        return delta_W

    def initialize_weights(self):
        n_input_units_bias = self.n_input_units + 1
        self.W = (np.random.rand(self.n_units, n_input_units_bias) - 0.5) * 2 / 100

    def predict(self, X):
        o = bias(X).dot(self.W.T)
        #o = self.W.dot(bias(X).T).T
        self.Z = sigmoid(o) if self.apply_sigmoid else o
        return self.Z
