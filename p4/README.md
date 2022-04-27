# Peter Rasmussen, Programming Assignment 4

This Python 3 program trains the k nearest neighbor predictor across six datasets, using accuracy for the classification data and normalized interquantile root mean squared error for the regression data.

## Getting Started

The package is designed to be executed as a module from the command line. The user must specify the
 output directory as illustrated below. The PRasmussenAlgospa2/resources
directory provides example output files - which echo the dynamically-generated input - for the user.

```shell
python -m path/to/p4  -i path/to/in_dir -o path/to/out_dir/ -k <folds> -v <val frac> -r <random state>
```

As an example:
```shell
python -m path/to/p4  -i path/to/in_dir -o path/to/out_dir/ -k 5 -v 0.1 -r 777
```

A summary of the command line arguments is below.

Positional arguments:

    -i, --src_dir               Input directory
    -o, --dst_dir               Output directory

Optional arguments:    

    -h, --help                 Show this help message and exit
    -k, --k_folds              Number of folds
    -v, --val_frac             Fraction of validation observations
    -r, --random_state         Provide pseudo-random seed

## Key parts of program
* run.py: Executes data loading, preprocessing, training, socring, and output creation.
* preprocessor.py: Preprocesses data: loading, imputation, discretization, and fold assignment.

* knn_classifier.py
  * Employs majority rule to predict
  * Scored on the basis of accuracy
* knn_regressor.py
  * Uses a gaussian kernel to weight nearest neighbor label values
  * Regression datasets are scored on the basis of normalized interquantile root mean squared error

## Features

* Performance metrics for each run for each dataset.
* Support for edited and condensed mode.
* Outputs provided as three files: 1) testing results, tuning results, and summary scores.
* Control over number of folds, validation fraction, and randomization.

## Output Files

See the ```testing_results.csv```, ```tuning_results.csv```, and summary.csv``` files in the ```data/``` directory.

## Licensing

This project is licensed under the CC0 1.0 Universal license.
