#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, classification_perceptron.py

This module provides various functions used in the classification perceptron routine.

"""
# Third party libraries
import numpy as np


def gradient(eta, Y, Yhat, X):
    return eta * (Y - Yhat).T.dot(X)


def predict_output(w, X):
    return w.dot(X.T).T


def normalize_output(output):
    return (np.exp(output.T) / np.sum(np.exp(output.T), axis=0)).T


def predict(w, X):
    output = predict_output(w, X)
    normed_output = normalize_output(output)
    return normed_output


