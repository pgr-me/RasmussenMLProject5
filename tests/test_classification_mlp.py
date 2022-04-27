#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, test_classification_mlp.py

"""
# Standard library imports
import json
from pathlib import Path
import warnings

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from p4.preprocessing import Preprocessor
from p4.mlp.mlp import MLP
from p4.preprocessing.split import make_splits
from p4.preprocessing.standardization import get_standardization_params, standardize, get_standardization_cols
from p4.utils import accuracy, dummy_categorical_label
from p4.mlp.layer import Layer

warnings.filterwarnings('ignore')

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


def test_regression_mlp():
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
        # Assign folds
        data["fold"] = make_splits(data, problem_class, label, k_folds=K_FOLDS, val_frac=None)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prep labels
        if problem_class == "classification":
            data[label] = data[label].astype(int)
            classes = sorted(data[label].unique())
            K = len(classes)
            # Dummy labels and save for later
            dummied_label_df, dummied_label_cols = dummy_categorical_label(data, label)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Validate: Iterate over each fold-run
        print(f"\tValidate")
        val_results_li = []
        test_sets = {}
        te_results_li = []

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

            # Save test data for later
            test_sets[fold] = dict(Y_te=Y_te, X_te=X_te, Y_tr=Y_tr, X_tr=X_tr)  # Save test for later

            D = X_val.shape[1]

            n_runs = 150
            hidden_units_li = [[10, 10], [10, 8], [8, 6], [10, 6], [6, 6], [6, 4], [4, 2], [3, 2]]
            val_results = []
            for eta in [0.1, 0.2, 0.4, 1]:
            #for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 1]:
                for hidden_units in hidden_units_li:
                    h1, h2 = hidden_units

                    # Define layers
                    layers = [Layer("input", D, n_input_units=D, apply_sigmoid=True),
                              Layer("hidden_1", h1, n_input_units=None, apply_sigmoid=True),
                              Layer("hidden_2", h2, n_input_units=None, apply_sigmoid=True),
                              Layer("output", K, n_input_units=None, apply_sigmoid=True)
                              ]
                    mlp = MLP(layers, D, eta, problem_class, n_runs=n_runs)
                    mlp.initialize_weights()

                    # Train MLP
                    mlp.train(Y_tr, X_tr, Y_val, X_val)
                    index = ["eta", "h1", "h2"]
                    outputs = pd.DataFrame([[x] * n_runs for x in [eta, h1, h2]], index=index).transpose()
                    outputs["ce_val"] = mlp.val_scores
                    outputs["acc_val"] = mlp.val_acc
                    outputs["run"] = range(n_runs)
                    val_results.append(outputs)
            val_results = pd.concat(val_results)
            val_results["fold"] = fold
            val_results = val_results.set_index(["fold", "run"]).reset_index()
            val_results_li.append(val_results)

        val_results = pd.concat(val_results_li)
        group = ["eta", "h1", "h2"]
        subset = ["fold", "eta", "h1", "h2"]
        val_summary = val_results.copy().sort_values(by="ce_val").drop_duplicates(subset=subset)
        val_summary = val_summary.groupby(group).mean().sort_values(by="ce_val")
        best_val_params = val_summary.reset_index().iloc[0].to_dict()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Test
        print(f"\tTest")
        eta = best_val_params["eta"]
        h1 = int(best_val_params["h1"])
        h2 = int(best_val_params["h2"])
        n_runs = int(best_val_params["run"])
        for fold in range(1, K_FOLDS + 1):
            print(f"\t\t{fold}")
            test_set = test_sets[fold]
            Y_tr, X_tr = test_set["Y_tr"], test_set["X_tr"]
            Y_te, X_te = test_set["Y_te"], test_set["X_te"]
            layers = [Layer("input", D, n_input_units=D, apply_sigmoid=True),
                      Layer("hidden_1", h1, n_input_units=None, apply_sigmoid=True),
                      Layer("hidden_2", h2, n_input_units=None, apply_sigmoid=True),
                      Layer("output", K, n_input_units=None, apply_sigmoid=True)
                      ]

            mlp = MLP(layers, D, eta, problem_class, n_runs=n_runs)
            mlp.initialize_weights()
            mlp.train(Y_tr, X_tr, Y_te, X_te)

            index = ["eta", "h1", "h2"]
            outputs = pd.DataFrame([[x] * n_runs for x in [eta, h1, h2]], index=index).transpose()
            outputs["ce_te"] = mlp.val_scores
            outputs["acc_te"] = accuracy(Y_te, mlp.predict(X_te))
            outputs["run"] = range(n_runs)
            outputs["fold"] = fold
            te_results_li.append(outputs)

        te_results = pd.concat(te_results_li).set_index(["fold", "run"]).reset_index()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save outputs
        print("\tSave")
        te_results_dst = DST_DIR / f"mlp_{dataset}_te_results.csv"
        val_results_dst = DST_DIR / f"mlp_{dataset}_val_results.csv"
        val_summary_dst = DST_DIR / f"mlp_{dataset}_val_summary.csv"

        te_results.to_csv(te_results_dst, index=False)
        val_results.to_csv(val_results_dst, index=False)
        val_summary.to_csv(val_summary_dst)


if __name__ == "__main__":
    test_regression_mlp()
