#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 4, test_sarsa.py

"""
# Standard library imports
import collections as c
from copy import deepcopy
import json
from pathlib import Path
import warnings

# Third party imports
import numpy as np
import pandas as pd
from numba import jit, njit
import numba as nb

# Local imports
from p5.q_learning import Preprocessor
from p5.sarsa.regression_perceptron import predict, train_perceptron
from p5.utils import mse
from p5.q_learning.split import make_splits
from p5.q_learning.standardization import get_standardization_params, standardize, get_standardization_cols

warnings.filterwarnings('ignore')

# Define constants
TEST_DIR = Path(".").absolute()
REPO_DIR = TEST_DIR.parent
P4_DIR = REPO_DIR / "p5"
SRC_DIR = REPO_DIR / "data"
DST_DIR = REPO_DIR / "data" / "out"
DST_DIR.mkdir(exist_ok=True, parents=True)
THRESH = 0.01
K_FOLDS = 5
VAL_FRAC = 0.2

# Load data catalog and tuning params
with open(SRC_DIR / "data_catalog.json", "r") as file:
    data_catalog = json.load(file)
data_catalog = {k: v for k, v in data_catalog.items() if k in ["forestfires", "machine", "abalone"]}


def test_regression_perceptron():
    # Iterate over each dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for dataset_name, dataset_meta in data_catalog.items():
        print(dataset_name)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Preprocess dataset
        preprocessor = Preprocessor(dataset_name, dataset_meta, SRC_DIR)
        preprocessor.load()
        preprocessor.drop()
        preprocessor.identify_features_label_id()
        preprocessor.replace()
        preprocessor.log_transform()
        preprocessor.set_data_classes()
        preprocessor.impute()
        preprocessor.dummy()
        preprocessor.set_data_classes()
        preprocessor.shuffle()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract feature and label columns
        label = preprocessor.label
        features = [x for x in preprocessor.features if x != label]
        problem_class = dataset_meta["problem_class"]
        data = preprocessor.data.copy()
        data = data[[label] + features]
        if problem_class == "classification":
            data[label] = data[label].astype(int)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assign folds
        data["fold"] = make_splits(data, problem_class, label, k_folds=K_FOLDS, val_frac=None)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Validate: Iterate over each fold-run
        print(f"\tValidate")
        val_results_li = []
        test_sets = {}
        etas = {}
        te_results_li = []
        for fold in range(1, K_FOLDS + 1):
            print(f"\t\t{fold}")
            test_mask = data["fold"] == fold
            test = data.copy()[test_mask].drop(axis=1, labels="fold")  # We'll save the test for use later
            train_val = data.copy()[~test_mask].drop(axis=1, labels="fold")
            train_val["train"] = make_splits(train_val, problem_class, label, k_folds=None, val_frac=VAL_FRAC)
            train_mask = train_val["train"] == 1
            train = train_val.copy()[train_mask].drop(axis=1, labels="train")
            val = train_val.copy()[~train_mask].drop(axis=1, labels="train")

            # Get standardization parameters from training-validation set
            cols = get_standardization_cols(train, features)
            means, std_devs = get_standardization_params(train.copy()[cols])

            # Standardize data
            train = train.drop(axis=1, labels=cols).join(standardize(train[cols], means, std_devs))
            val = val.drop(axis=1, labels=cols).join(standardize(val[cols], means, std_devs))
            test = test.drop(axis=1, labels=cols).join(standardize(test[cols], means, std_devs))  # Save test for later

            # Add bias terms
            train["intercept"] = 1
            val["intercept"] = 1
            test["intercept"] = 1  # Save test for later

            YX_tr = train.copy().astype(np.float64).values
            YX_te = test.copy().astype(np.float64).values  # Save test for later
            YX_val = val.copy().astype(np.float64).values
            Y_tr, X_tr = YX_tr[:, 0].reshape(len(YX_tr), 1), YX_tr[:, 1:]
            test_sets[fold] = dict(Y_te=YX_te[:, 0].reshape(len(YX_te), 1), X_te=YX_te[:, 1:])  # Save test for later
            Y_val, X_val = YX_val[:, 0].reshape(len(YX_val), 1), YX_val[:, 1:]
            for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 1]:
                w_tr = train_perceptron(Y_tr, X_tr, eta, thresh=THRESH)
                Yhat_val = predict(X_val, w_tr)
                mse_val = mse(Y_val, Yhat_val)
                val_results_li.append(dict(dataset_name=dataset_name, fold=fold, eta=eta, mse_val=mse_val))
                etas[(fold, eta)] = w_tr  # Save etas for later
        val_results = pd.DataFrame(val_results_li)
        val_summary = val_results.groupby("eta")["mse_val"].mean().sort_values().to_frame()
        best_eta = val_summary.index.values[0]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Test
        print(f"\tTest")
        for fold in range(1, K_FOLDS + 1):
            print(f"\t\t{fold}")
            w_tr = etas[(fold, best_eta)]
            Y_te, X_te = test_sets[fold]["Y_te"], test_sets[fold]["X_te"]
            Yhat_te = predict(X_te, w_tr)
            mse_te = mse(Y_te, Yhat_te)
            te_results_li.append(dict(problem_class=problem_class, dataset_name=dataset_name, fold=fold, mse_te=mse_te,
                                      best_eta=best_eta))
        te_results = pd.DataFrame(te_results_li)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save outputs
        print("\tSave")
        te_results_dst = DST_DIR / f"perceptron_{dataset_name}_te_results.csv"
        val_results_dst = DST_DIR / f"perceptron_{dataset_name}_val_results.csv"
        val_summary_dst = DST_DIR / f"perceptron_{dataset_name}_val_summary.csv"

        te_results.to_csv(te_results_dst, index=False)
        val_results.to_csv(val_results_dst, index=False)
        val_summary.to_csv(val_summary_dst)


if __name__ == "__main__":
    test_regression_perceptron()
