"""Utility functions that provide the absolute path to project directories.

Typical usage example:

# Path to the checkpoint directory used to store intermediate training
# checkpoints for experiment name stored in `experiment_name`.
checkpointsdir(experiment_name)
"""

import git
import shutil
import os
from typing import Optional
import subprocess


def gitdir() -> str:
    """Find the absolute path to the GitHub repository root.
    """
    git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
    git_root = git_repo.git.rev_parse('--show-toplevel')
    return git_root


def datadir(path: str, mkdir: Optional[bool] = True) -> str:
    """The absolute path to the data directory.

    Data directory is for training and testing data. Here the path is created
    if it does not exist upon call if `mkdir` is True.

    Args:
        path: A string for directory name located at the data directory. mkdir:
        An optional boolean for whether to create the directory if it
            does not exist.
    """
    path = os.path.join(gitdir(), 'data/', path)
    if (not os.path.exists(path)) and mkdir:
        os.makedirs(path)
    return path


def plotsdir(path: str, mkdir: Optional[bool] = True) -> str:
    """The absolute path to the plot directory.

    Plot directory stores figure of experiment results. Here the path is
    created if it does not exist upon call if `mkdir` is True.

    Args:
        path: A string for directory name located at the plot directory. mkdir:
        An optional boolean for whether to create the directory if it
            does not exist.
    """
    path = os.path.join(gitdir(), 'plots/', path)
    if (not os.path.exists(path)) and mkdir:
        os.makedirs(path)
    return path


def checkpointsdir(path: str, mkdir: Optional[bool] = True) -> str:
    """The absolute path to the checkpoint directory.

    Checkpoint directory stores intermediate training checkpoints, e.g.,
    network weights. Here the path is created if it does not exist upon call if
    `mkdir` is True.

    Args:
        path: A string for directory name located at the checkpoint directory.
        mkdir: An optional boolean for whether to create the directory if it
            does not exist.
    """
    path = os.path.join(datadir('checkpoints'), path)
    if (not os.path.exists(path)) and mkdir:
        os.makedirs(path)
    return path


def logsdir(path: str, mkdir: Optional[bool] = True) -> str:
    """The absolute path to the logs directory.

    Logs directory stores Tensorboard logs. Here the path is created if it does
    not exist upon call if `mkdir` is True.

    Args:
        path: A string for directory name located at the checkpoint directory.
        mkdir: An optional boolean for whether to create the directory if it
            does not exist.
    """
    path = os.path.join(datadir('logs'), path)
    if (not os.path.exists(path)) and mkdir:
        os.makedirs(path)
    return path


def configsdir(mkdir: Optional[bool] = True) -> str:
    """The absolute path to the configs directory.

    Configurations directory, stores the default hyperparameter values for
    various experiments.

    Args:
        mkdir: An optional boolean for whether to create the directory if it
            does not exist.
    """
    path = os.path.join(gitdir(), 'configs')
    if (not os.path.exists(path)) and mkdir:
        os.makedirs(path)
    return path


def rm_cachedir(wildcard, path=os.path.join(gitdir(), 'srcsep/_cached_dir')):
    """The absolute path to the configs directory.

    Configurations directory, stores the default hyperparameter values for
    various experiments.

    Args:
        mkdir: An optional boolean for whether to create the directory if it
            does not exist.
    """
    command = ("find " + path + " -type d -name " + wildcard +
               " -exec rm -rf {} +")
    subprocess.Popen(command.split())