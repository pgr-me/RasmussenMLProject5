# Peter Rasmussen, Programming Assignment 5

This Python 3 program trains three reinforcement learners across three larger track datasets and two toy / test
datasets. We use three different reinforcement learning algorithms: value iteration, Q learning, and SARSA. For the Q
learning and SARSA algorithms, the user can select among two policies: `softmax` and `greedy-epsilon`. We employ two
penalties in separate runs for running into a wall: `stay-in-place` and `back-to-beginning`.

## Getting Started

The package is designed to be executed as a module from the command line. The user must specify the output directory as
illustrated below. The `RasmussenMLProject5/data/out` directory provides example output files - which echo the
dynamically-generated input - for the user.

```shell
python -m path/to/p5  -i path/to/in_dir -o path/to/out_dir/
```

A summary of the command line arguments is below.

Positional arguments:

    -i, --src_dir               Input directory
    -o, --dst_dir               Output directory

Optional arguments:

    -h, --help                 Show this help message and exit

## Key parts of program

* run.py: Executes data loading, track preparation, training, socring, and output creation. This module calls functions
  from the following sub-modules:
    * p5/run/run_value_iteration.py
    * p5/run/run_q_learning.py
    * p5/run/run_sarsa.py

## Features

* Support for three reinforcement learning algorithms across two penalty scenarios and two policies
* Progress and performance metrics for each run for each dataset.
* Control over input and output directory.

## Output Files

See the `testing` directory for example outputs.

## Licensing

This project is licensed under the CC0 1.0 Universal license.
