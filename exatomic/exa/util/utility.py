# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Utilities
#####################
Commonly used functions (primarily for convenience and repetition reduction).
"""
import os
import sys
import numpy as np
from datetime import datetime


sep2 = os.sep + os.sep
sizes = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']


def datetime_header(title=''):
    """
    Creates a simple header string containing the current date/time stamp
    delimited using "=".
    """
    return '\n'.join(('=' * 80, title + ': ' + str(datetime.now()), '=' * 80))


def mkp(*args, mk=True):
    """
    Generate a directory path, and create it if requested.

    .. code-block:: Python

        filepath = mkp('base', 'folder', 'file')
        dirpath = mkp('root', 'path', 'folder', mk=True)

    Args:
        args: File or directory path segments to be concatenated
        mk (bool): Make the directory (if it doesn't exist)

    Returns:
        path (str): File or directory path
    """
    path = os.sep.join(list(args))
    if mk:
        while sep2 in path:
            path = path.replace(sep2, os.sep)
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    return path


def convert_bytes(value):
    """
    Reduces bytes to more convenient units (i.e. KiB, GiB, TiB, etc.).

    Args:
        values (int): Value in Bytes

    Returns:
        tup (tuple): Tuple of value, unit (e.g. (10, 'MiB'))
    """
    n = np.rint(len(str(value))/4).astype(int)
    return value/(1024**n), sizes[n]


def get_internal_modules(key='exa'):
    """
    Get a list of modules belonging to the given package.

    Args:
        key (str): Package or library name (e.g. "exa")
    """
    key += '.'
    return [v for k, v in sys.modules.items() if k.startswith(key)]

