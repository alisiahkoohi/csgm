import itertools
import os
import subprocess

from .project_path import configsdir
from .config import parse_input_args, read_config


def make_complete_args(config_file, **kwargs):
    """Make the arguments for the query."""

    args = read_config(os.path.join(configsdir(), config_file))
    args = parse_input_args(args)

    # Create args from the kwargs.
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def query_experiments(config_file, **kwargs):
    """Make the arguments for the query."""
    args_list = []
    key_list = []
    # Create args from the kwargs.
    for key, value in kwargs.items():
        if (not isinstance(value, list)) and (not isinstance(value, tuple)):
            value = (value, )
        args_list.append(value)
        key_list.append(key)

    list_args = []
    for args in itertools.product(*args_list):
        list_args.append(dict(zip(key_list, args)))

    experiment_args = []
    for kwargs in list_args:
        args = make_complete_args(config_file, **kwargs)
        experiment_args.append(args)

    if not experiment_args:
        args = read_config(os.path.join(configsdir(), config_file))
        args = parse_input_args(args)
        experiment_args = [args]

    return experiment_args

