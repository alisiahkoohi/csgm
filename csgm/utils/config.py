import argparse
import json


def read_config(filename):
    """Read input variables and values from a json file."""
    with open(filename) as f:
        configs = json.load(f)
    return configs


def parse_input_args(args):
    "Use variables in args to create command line input parser."
    parser = argparse.ArgumentParser(description='')
    for key, value in args.items():
        parser.add_argument('--' + key, default=value, type=type(value))
    parsed_args = parser.parse_args()
    return parsed_args


def make_experiment_name(args):
    """Make experiment name based on input arguments"""
    experiment_name = args.experiment_name + '_'
    for key, value in vars(args).items():
        if key not in [
                'experiment_name', 'cuda', 'seed', 'save_freq', 'phase',
                'val_batchsize', 'input_size', 'testing_epoch',
                'testing_nsamples', 'test_idx'
        ]:
            experiment_name += key + '-{}_'.format(value)
    return experiment_name[:-1].replace('[', '').replace(']', '').replace(
        ',', '-').replace(' ', '')
