"""Peter Rasmussen, Programming Assignment 5, run.py

The run function ingests user inputs to train reinforcement learners using value iteration, Q learning and SARSA.

Outputs are saved to the user-specified directory.

"""

# Standard library imports
import logging
import os
from pathlib import Path

# Local imports
from p5.run.run_q_learning import run_q_learning
from p5.run.run_sarsa import run_sarsa
from p5.run.run_value_iteration import run_value_iteration


def run(
        src_dir: Path,
        dst_dir: Path
):
    """
    Train and score a majority predictor across six datasets.
    :param src_dir: Input directory that provides each dataset and params files
    :param dst_dir: Output directory
    """
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    log_path = dir_path / "p5.log"
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format=log_format)

    logging.debug(f"Begin: src_dir={src_dir.name}, dst_dir={dst_dir.name}.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logging.debug("Run value iteration")
    run_value_iteration(src_dir, dst_dir)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logging.debug("Run value iteration")
    run_q_learning(src_dir, dst_dir)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logging.debug("Run SARSA")
    run_sarsa(src_dir, dst_dir)

    logging.debug("Finish.\n")
