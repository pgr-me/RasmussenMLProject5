"""Peter Rasmussen, Programming Assignment 4, run.py

The run function ingests user inputs to train majority predictors on six different datasets.

Outputs are saved to the user-specified directory.

"""

# Standard library imports
from collections import defaultdict
import json
import logging
import os
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from p5.q_learning_old import Preprocessor, get_standardization_cols, get_standardization_params, standardize
from p5.q_learning_old.split import make_splits


def run(
        src_dir: Path,
        dst_dir: Path,
        k_folds: int,
        val_frac: float,
        random_state: int,
):
    """
    Train and score a majority predictor across six datasets.
    :param src_dir: Input directory that provides each dataset and params files
    :param dst_dir: Output directory
    :param k_folds: Number of folds to partition the data into
    :param val_frac: Validation fraction of train-validation set
    :param random_state: Random number seed

    """
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    log_path = dir_path / "p5.log"
    with open(src_dir / "discretize.json") as file:
        discretize_dicts = json.load(file)
    discretize_dicts = defaultdict(lambda: {}, discretize_dicts)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format=log_format)

    logging.debug(f"Begin: src_dir={src_dir.name}, dst_dir={dst_dir.name}, seed={random_state}.")

    # Load data catalog and tuning params
    with open(src_dir / "data_catalog.json", "r") as file:
        data_catalog = json.load(file)
    with open(src_dir / "tuning_params.json", "r") as file:
        tuning_params = json.load(file)

    # Initialize tuning results and output lists
    # tuning_results = []
    output = []
    best_params = []
    testing_results = []

    # Loop over each dataset and its metadata using the data_catalog
    tuning_results_li = []

    # Create randomized grid of parameters
    runs = {}
    for i in range(4):
        runs[i] = {"k": np.random.choice(tuning_params["ks"]),
                   "sigma": np.random.choice(tuning_params["sigmas"]),
                   "eps": np.random.choice(tuning_params["epsilons"])}
    # data_catalog = {k: v for k, v in data_catalog.items() if k=="car"}
    # Iterate over each dataset
    for dataset_name, dataset_meta in data_catalog.items():

        # if dataset_name == "abalone":
        print(f"Dataset: {dataset_name}")
        tuning_results = []

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.debug(f"Load and process dataset {dataset_name}.")

        # Load data: Set column names, data types, and replace values
        preprocessor = Preprocessor(dataset_name, dataset_meta, src_dir)
        preprocessor.load()

        # Identify which columns are features, which is the label, and any ID columns
        preprocessor.identify_features_label_id()

        # Replace values: Ordinal strings (lower, higher) replace with numeric values
        preprocessor.replace()

        # Log transform indicated columns (default is to take selected columns from dataset_meta)
        preprocessor.log_transform()

        # Impute missing values
        preprocessor.impute()

        # Dummy categorical columns
        preprocessor.dummy()

        # Discretize indicated columns
        preprocessor.discretize(discretize_dicts[dataset_name])

        # Randomize the order of the data
        preprocessor.shuffle(random_state=random_state)

        # Extract dataframe from preprocessor object
        data = preprocessor.data.copy()

        # Define each column as a feature, label, or index
        feature_cols = preprocessor.features
        label_col = preprocessor.label
        problem_class = dataset_meta["problem_class"]  # regression or classification
        index = preprocessor.index
        if problem_class == "classification":
            data[label_col] = data[label_col].astype(int)

        # Split off validation samples from train / test
        splits = make_splits(data, problem_class, label_col, k_folds=None, val_frac=val_frac)
        mask = splits["train"] == 0
        val = splits.copy()[mask].join(data).drop(axis=1, labels="train")
        train_test = splits.copy()[~mask].join(data).drop(axis=1, labels="train")

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.debug(f"Tune {dataset_name}.")

        # Make folds for validation set
        folds = make_splits(val, problem_class, label_col, k_folds=k_folds, val_frac=None)
        val = folds.join(val)

        # Iterate over each fold
        for fold in range(1, k_folds + 1):

            print(f"\tFold: {fold}")
            # Split test and train-validation sets
            mask = val["fold"] == fold
            val_test = val.copy()[mask]
            val_train = val.copy()[~mask]

            # Get standardization parameters from training-validation set
            cols = get_standardization_cols(val_train, feature_cols)
            means, std_devs = get_standardization_params(val_train.copy()[cols])

            # Standardize data
            val_test = val_test.drop(axis=1, labels=cols).join(standardize(val_test[cols], means, std_devs))
            val_train = val_train.drop(axis=1, labels=cols).join(standardize(val_train[cols], means, std_devs))

            # Make the label the first column
            val_test = val_test.reset_index().set_index(label_col).reset_index().set_index(index)
            val_train = val_train.reset_index().set_index(label_col).reset_index().set_index(index)

            for method in tuning_params["methods"]:
                for run_id, run_di in runs.items():
                    k, sigma, eps = run_di["k"], run_di["sigma"], run_di["eps"]
                    if problem_class == "regression":
                        metric = "rmseiqr"
                        params = ["method", "k", "sigma", "epsilon"]
                        knn = KNNRegressor(val_train, k, label_col, index, method=method)
                        knn.make_lookup_table()
                        knn.train_classification_perceptron(sigma, eps)
                        pred = knn.predict(val_test, sigma)
                        results = knn.make_results(val_test, pred)
                        scores = knn.score(results)
                        scores.update(k=k, method=method, sigma=sigma, epsilon=eps, problem_class=problem_class,
                                      dataset_name=dataset_name, fold=fold)
                        tuning_results.append(scores)

                    else:
                        metric = "acc"
                        params = ["method", "k"]
                        knn = KNNClassifier(val_train, k, label_col, index, method=method)
                        knn.make_lookup_table()
                        knn.train_classification_perceptron()

                        pred = knn.predict(val_test)

                        results = knn.make_results(val_test, pred)
                        scores = knn.score(results).to_dict()[0]
                        scores.update(k=k, method=method, sigma=None, epsilon=None, problem_class=problem_class,
                                      dataset_name=dataset_name, fold=fold)
                        tuning_results.append(scores)

        # Organize tuning results and select best params across folds
        cols = ["dataset_name", "problem_class", "fold", "k", "method", "sigma", "epsilon"]
        cols += [x for x in scores.keys() if x not in cols]

        tuning_results = pd.DataFrame(tuning_results)[cols].sort_values(by=metric, ascending=False)
        tuning_results[metric] = tuning_results[metric].fillna(0)
        tuning_results["method"] = tuning_results["method"].astype(str)
        tuning_results_li.append(tuning_results)

        # Get best params by averaging across folds
        best = tuning_results.copy().groupby(params)[metric].mean()

        best = best.sort_values(ascending=False).head(1).reset_index().loc[0].to_dict()
        best.update({"dataset_name": dataset_name, "problem_class": problem_class})
        best_params.append(best)
        best = defaultdict(lambda: None, best)
        method, k, sigma, eps = best["method"], best["k"], best["sigma"], best["epsilon"]

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        logging.debug(f"Train / test {dataset_name}.")

        # Make folds for validation set
        folds = make_splits(train_test, problem_class, label_col, k_folds=k_folds, val_frac=None)
        train_test = folds.join(train_test)

        # Iterate over each fold
        for fold in range(1, k_folds + 1):

            # Split test and train-validation sets
            mask = train_test["fold"] == fold
            test = train_test.copy()[mask]
            train = train_test.copy()[~mask]

            # Get standardization parameters from training-validation set
            cols = get_standardization_cols(train, feature_cols)
            means, std_devs = get_standardization_params(train.copy()[cols])

            # Standardize data
            test = test.drop(axis=1, labels=cols).join(standardize(test[cols], means, std_devs))
            train = train.drop(axis=1, labels=cols).join(standardize(train[cols], means, std_devs))

            # Make the label the first column
            test = test.reset_index().set_index(label_col).reset_index().set_index(index)
            train = train.reset_index().set_index(label_col).reset_index().set_index(index)

            if problem_class == "regression":
                metric = "rmseiqr"
                params = ["method", "k", "sigma", "epsilon"]
                knn = KNNRegressor(train, k, label_col, index, method=method)
                knn.make_lookup_table()
                if len(knn.all_lookups) > 0:
                    knn.train_classification_perceptron(sigma, eps)
                    pred = knn.predict(test, sigma)
                    results = knn.make_results(test, pred)
                    scores = knn.score(results)
                else:
                    scores = dict(sse=0, rmse=0, nrmse=0, rmseiqr=0)
                scores.update(k=k, method=method, sigma=sigma, epsilon=eps, problem_class=problem_class,
                              dataset_name=dataset_name, fold=fold)


            else:
                metric = "acc"
                params = ["method", "k"]
                knn = KNNClassifier(val_train, k, label_col, index, method=method)
                knn.make_lookup_table()
                knn.train_classification_perceptron()

                pred = knn.predict(val_test)
                results = knn.make_results(val_test, pred)
                scores = knn.score(results).to_dict()[0]
                scores.update(k=k, method=method, sigma=None, epsilon=None, problem_class=problem_class,
                              dataset_name=dataset_name, fold=fold)
            testing_results.append(scores)
            pd.DataFrame(testing_results).to_csv("testing_results.csv")
            pd.DataFrame(best_params).set_index(["dataset_name", "problem_class"]).to_csv("best_params.csv")

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    logging.debug("Process outputs.")
    testing_results = pd.DataFrame(testing_results)
    testing_results[["sigma", "epsilon"]] = testing_results[["sigma", "epsilon"]].astype(str)
    gp = ["dataset_name", "problem_class", "k", "method", "sigma", "epsilon"]
    summary_results = testing_results.drop(axis=1, labels="fold").groupby(gp).mean()
    best_params = pd.DataFrame(best_params).set_index(["dataset_name", "problem_class"])

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    logging.debug("Save outputs.")
    tuning_results_dst = dst_dir / "tuning_results.csv"
    testing_results_dst = dst_dir / "testing_results.csv"
    best_params_dst = dst_dir / "best_params.csv"
    summary_dst = dst_dir / "summary.csv"

    tuning_results.to_csv(tuning_results_dst, index=False)
    testing_results.to_csv(testing_results_dst, index=False)
    best_params.to_csv(best_params_dst)
    summary_results.to_csv(summary_dst)

    logging.debug("Finish.\n")


