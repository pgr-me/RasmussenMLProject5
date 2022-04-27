#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, test_nodes.py

This module tests the nodes module.

K-Folds cross validation strategy:
    Each fold-run is its own experiment
    Assign each observation to one of five folds

    # Do validation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For fold i:
        fold i is test
        fold ~i is train-val
        Split train-val into train and val (80 / 20)
        Train on train
        Predict trained model using different param sets on val
    Take best params over all fold i's: Take a mean to determine best params

    # Do testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For fold i:
         ...
        Test on best params

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
from p4.preprocessing import Preprocessor
from p4.perceptrons.regression_perceptron import predict, train_perceptron
from p4.utils import mse, cross_entropy, dummy_categorical_label, accuracy
from p4.preprocessing.split import make_splits
from p4.preprocessing.standardization import get_standardization_params, standardize, get_standardization_cols
from p4.perceptrons.classification_perceptron import gradient, \
    normalize_output, predict_output, predict

warnings.filterwarnings('ignore')

rng = np.random.default_rng()

# Define constants
TEST_DIR = Path(".").absolute()
REPO_DIR = TEST_DIR.parent
P4_DIR = REPO_DIR / "p4"
SRC_DIR = REPO_DIR / "data"
DST_DIR = REPO_DIR / "data" / "out"
DST_DIR.mkdir(exist_ok=True, parents=True)
THRESH = 0.01
K_FOLDS = 5
VAL_FRAC = 0.2

# Load data catalog and tuning params
with open(SRC_DIR / "data_catalog.json", "r") as file:
    data_catalog = json.load(file)
data_catalog = {k: v for k, v in data_catalog.items() if k in ["breast-cancer-wisconsin", "car", "house-votes-84"]}
#data_catalog = {k: v for k, v in data_catalog.items() if k == "car"}


def test_classification_perceptron():
    # Iterate over each dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for dataset, dataset_meta in data_catalog.items():
        print(dataset)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Preprocess dataset
        preprocessor = Preprocessor(dataset, dataset_meta, SRC_DIR)
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Dummy labels and save for later
        dummied_label_df, dummied_label_cols = dummy_categorical_label(data, label)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set bias / intercept
        data["intercept"] = 1
        d = len(features) + 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assign folds
        data["fold"] = make_splits(data, problem_class, label, k_folds=K_FOLDS, val_frac=None)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Dummy labels
        if problem_class == "classification":
            data[label] = data[label].astype(int)
            classes = sorted(data[label].unique())
            K = len(classes)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Validate: Iterate over each fold-run
        print(f"\tValidate")
        test_sets = {}
        te_results_li = []
        val_results = []
        w_trs = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Validation: Iterate over each fold
        for fold in range(1, K_FOLDS + 1):
            print(f"\t\t{fold}")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Split test, train, and validation
            test_mask = data["fold"] == fold
            test = data.copy()[test_mask].drop(axis=1, labels="fold")  # We'll save the test for use later
            train_val = data.copy()[~test_mask].drop(axis=1, labels="fold")
            train_val["train"] = make_splits(train_val, problem_class, label, k_folds=None, val_frac=VAL_FRAC)
            train_mask = train_val["train"] == 1
            train = train_val.copy()[train_mask].drop(axis=1, labels="train")
            val = train_val.copy()[~train_mask].drop(axis=1, labels="train")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get standardization parameters from training-validation set
            cols = get_standardization_cols(train, features)
            means, std_devs = get_standardization_params(train.copy()[cols])

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Standardize data
            X_tr_df = train.drop(axis=1, labels=[label] + cols).join(standardize(train[cols], means, std_devs))
            X_val_df = val.drop(axis=1, labels=[label] + cols).join(standardize(val[cols], means, std_devs))
            X_te_df = test.drop(axis=1, labels=[label] + cols).join(standardize(test[cols], means, std_devs))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Convert train, test, and validation dataframes into arrays
            X_tr = X_tr_df.copy().astype(np.float64).values
            X_val = X_val_df.copy().astype(np.float64).values
            X_te = X_te_df.copy().astype(np.float64).values

            Y_tr = dummied_label_df.loc[train.index].astype(np.float64).values
            Y_val = dummied_label_df.loc[val.index].astype(np.float64).values
            Y_te = dummied_label_df.loc[test.index].astype(np.float64).values

            test_sets[fold] = dict(X_te=X_te, Y_te=Y_te)  # Save test for later
            for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 1]:
                w_tr = (np.random.rand(K, d) - 0.5) * 2 / 100
                for i in range(500):
                    rnd_ix = rng.permuted(np.arange(len(Y_tr)))

                    # Compute training error
                    Yhat_tr = predict(w_tr, X_tr[rnd_ix, :])
                    w_tr = w_tr + gradient(eta, Y_tr[rnd_ix, :], Yhat_tr, X_tr[rnd_ix, :])

                    ce_tr = cross_entropy(Y_tr[rnd_ix, :], Yhat_tr)
                    # Compute cross entropy: pg 263
                    Yhat_val = predict(w_tr, X_val)
                    ce_val = cross_entropy(Y_val, Yhat_val)
                    acc_val = accuracy(Y_val, Yhat_val)
                    w_trs.append(dict(dataset=dataset, fold=fold, eta=eta, iteration=i, w_tr=w_tr))
                    val_results.append(
                        dict(dataset_name=dataset, fold=fold, iteration=i, eta=eta, ce_tr=ce_tr, ce_val=ce_val,
                             acc_val=acc_val))

        val_results = pd.DataFrame(val_results)
        subset = ["dataset_name", "fold", "eta"]
        val_summary = val_results.copy().dropna().sort_values(by="ce_val").drop_duplicates(subset=subset)
        mean_ce_vals = val_summary.groupby("eta")["ce_val"].mean().sort_values()
        median_iter = val_summary.groupby("eta")["iteration"].median().round().astype(int)
        best_eta = mean_ce_vals.index.values[0]
        best_eta_iter = median_iter.loc[best_eta]
        w_trs = pd.DataFrame(w_trs).set_index(["dataset", "fold", "eta", "iteration"])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Test
        print(f"\tTest")
        for fold in range(1, K_FOLDS + 1):
            print(f"\t\t{fold}")
            w_tr = w_trs.loc[dataset, fold, best_eta, best_eta_iter].loc["w_tr"]
            Y_te, X_te = test_sets[fold]["Y_te"], test_sets[fold]["X_te"]
            Yhat_te = predict(w_tr, X_te)
            ce_te = cross_entropy(Y_te, Yhat_te)
            acc_te = accuracy(Y_te, Yhat_te)
            di = dict(problem_class=problem_class, dataset_name=dataset, fold=fold, best_eta=best_eta,
                      best_eta_iter=best_eta_iter, ce_te=ce_te, acc_te=acc_te)
            te_results_li.append(di)
        te_results = pd.DataFrame(te_results_li)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save outputs
        print("\tSave")
        te_results_dst = DST_DIR / f"perceptron_{dataset}_te_results.csv"
        val_results_dst = DST_DIR / f"perceptron_{dataset}_val_results.csv"
        val_summary_dst = DST_DIR / f"perceptron_{dataset}_val_summary.csv"

        te_results.to_csv(te_results_dst, index=False)
        val_results.to_csv(val_results_dst, index=False)
        val_summary.to_csv(val_summary_dst, index=False)


if __name__ == "__main__":
    test_classification_perceptron()
