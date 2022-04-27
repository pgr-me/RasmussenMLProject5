#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, utils.py

This module provides miscellaneous utility functions that support the core perceptrons of this program.

"""
# Third party libraries
import numba as nb
import numpy as np
import pandas as pd
from numba import njit


def bias(X):
    bias_term = np.ones([len(X), 1])
    return np.hstack([X, bias_term])


def cross_entropy(Y: np.array, Yhat: np.array) -> float:
    return -np.sum(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))


@njit
def mse(Y, Yhat):
    return np.sum(np.square(np.subtract(Y, Yhat))) / len(Y)


def sigmoid(output):
    return 1 / (1 + np.exp(-output))


def sigmoid_update(weights):
    return weights * (1 - weights)


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


def dummy_categorical_label(data: pd.DataFrame, label: str) -> tuple:
    """
    Dummy categorical label.
    :param data: Dataframe
    :param label: Label column
    :return: Tuple of data with dummied labels and label cols
    """

    dummied_labels: pd.DataFrame = pd.get_dummies(data[label].astype("category"), prefix=label)
    # Sort dummied cols in ascending order
    label_cols = sorted(list(dummied_labels))
    return dummied_labels[label_cols], label_cols


def accuracy(Y, Yhat):
    maxes = np.broadcast_to(np.max(Yhat, axis=1), Yhat.T.shape).T
    pred = np.greater_equal(Yhat, maxes).astype(np.uint8)
    pred_correct = (Y == pred).sum(axis=1) == Y.shape[1]
    return np.sum(pred_correct) / len(pred_correct)
