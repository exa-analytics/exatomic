# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Basis Set Support
##########################
Specific functions for NWChem basis set support.
"""
import numpy as np


def cartesian_ordering_function(l):
    """
    Generates an array of coordinates corresponding to powers of x, y, and z
    in a set of linearly dependent cartesian basis functions, representative of
    spin angular momentum values for a given orbital angular momentum.
    """
    m = l + 1
    n = (m + 1) * m // 2
    values = np.empty((n, 4), dtype=np.int64)
    h = 0
    for i in range(l, -1, -1):
        for j in range(l, -1, -1):
            for k in range(l, -1, -1):
                if i + j + k == l:
                    values[h] = [l, i, j, k]
                    h += 1
    return values


def spherical_ordering_function(l):
    """
    Generates an array of spin angular momentum values corresponding to a given
    orbital angular momentum.
    """
    if l == 0:
        return np.array([0], dtype=np.int64)
    elif l == 1:
        return np.array([1, -1, 0], dtype=np.int64)
    return np.arange(-l, l+1, 1, dtype=np.int64)
