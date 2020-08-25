# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
QE Type Conversions
######################
"""
lengths = {'alat': 'alat', 'bohr': 'au', 'crystal': 'crystal',
           'angstrom': 'A', 'crystal_sg': 'crystal_sg'}


def to_qe_type(value):
    """
    Convert Python object to the (string) representation to be read in by QE.

    Args:
        obj: Python object to be converted

    Returns:
        conv_obj (str): String representation of converted Python object
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, bool):
        return '.true.' if value else '.false.'
    elif isinstance(value, int) or isinstance(value, float):
        return str(value)
    else:
        raise Exception('Unknown type {0} [{1}].'.format(type(value), value))


def to_py_type(value):
    """
    Convert qe string object to a standard Python object.

    Args:
        obj (str): QE string value

    Returns:
        conv_obj: Python typed object
    """
    value = value.strip()
    value = value.replace(',', '')
    is_int = None
    try:
        is_int = int(value)
    except:
        pass
    is_float = None
    try:
        v1 = value.replace('d', 'e')
        is_float = float(v1)
    except:
        pass
    if '.true.' == value:
        return True
    elif '.false.' == value:
        return False
    elif is_int:
        return is_int
    elif is_float:
        return is_float
    elif isinstance(value, str):
        return value
    else:
        raise Exception('Unknown type {0} [{1}].'.format(type(value), value))


def get_length(value):
    """
    """
    value = value.replace(')', '').replace('(', '')
    return lengths[value]
