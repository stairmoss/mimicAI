#!/usr/bin/env python3

"""Shared utility functions."""

import argparse
import random

import numpy as np
import torch


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def fix_random_seed(random_seed: int):
    """
    Set the same random seed for the libraries and modules.
    Includes the ``random`` module, numpy, and torch.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    rd = random.Random()
    rd.seed(random_seed)
