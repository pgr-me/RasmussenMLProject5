#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, regression_perceptron.py

This module provides various functions used in the regression perceptron routine.

"""
# Third party libraries
import numpy as np
from numba import njit
import numba as nb

from p5.utils import mse


@njit
def predict(X, w):
    return w.dot(X.T).T


@njit
def gradient(Y: nb.float64[:], X: nb.float64[:], w: nb.float64[:], index: int, eta: float):
    x = X[index, :]
    r = Y[index]
    y = w.dot(x)
    return eta * (r - y) * x


@njit
def shuffle_indices(n: int) -> nb.int64[:]:
    """
    Shuffle a zero-indexed index.
    :param n: Number of elements in array
    :return: Array of randomly ordered indices
    """
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices


#@njit
def train_perceptron(Y: nb.float64[:], X: nb.float64[:], eta: float = 0.1, thresh: float = 0.01, max_iter: int = 500):
    """
    Train online perceptron for regression.
    :param Y: Target
    :param X: Features
    :param eta: Learning rate
    :param thresh: MSE fractional improvement threshold
    :param max_iter: Number of iterations
    :return: Trained weights
    """
    w = (np.random.rand(1, X.shape[1]) - 0.5) * 2 / 100
    mse_ = np.abs(0.00001)
    iteration = 0
    while True:
        # Randomly shuffle indices
        indices = shuffle_indices(Y.shape[0])

        # Update weights
        for index in indices:
            w = w + gradient(Y, X, w, index, eta)

        # Compute MSE and improvement from the latest iteration
        Yhat = predict(X, w)
        new_mse = mse(Y, Yhat)
        delta = 1 - (mse_ - new_mse) / mse_
        mse_ = new_mse
        iteration = iteration + 1

        # Return if delta improvement attained or max iterations reached
        if (delta <= thresh) or (iteration >= max_iter):
            return w
