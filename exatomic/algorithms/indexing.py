# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Indexing
#######################
Algorithms for generating indices.
"""
import numpy as np
from numba import jit
from exatomic.base import nbche


@jit(nopython=True, nogil=True, cache=nbche)
def starts_count(starts, count):
    """
    Generate sequential indices (for 2 dimensions) from starting values and
    lengths (counts).

    .. math:: Python

    Args:
        starts (:class:`numpy.ndarray`): Array of starting points
        count (int): Integer count

    Returns:
        objs (tuple): Outer sequential index, inner sequential index, resulting indicies
    """
    n = len(starts) * count
    outer = np.empty((n, ), dtype=np.int64)
    inner = outer.copy()
    index = inner.copy()
    h = 0
    for i, start in enumerate(starts):
        stop = start + count
        for j, value in enumerate(range(start, stop)):
            outer[h] = i
            inner[h] = j
            index[h] = value
            h += 1
    return (outer, inner, index)


@jit(nopython=True, nogil=True, cache=nbche)
def starts_counts(starts, counts):
    """
    Generate a pseudo-sequential array from initial values and counts.

    Args:
        starts (array): Starting points for array generation
        counts (array): Values by which to increment from each starting point

    Returns:
        arrays (tuple): First index, second index, and indices to select, respectively
    """
    n = np.sum(counts)
    i_idx = np.empty((n, ), dtype=np.int64)
    j_idx = i_idx.copy()
    values = j_idx.copy()
    h = 0
    for i, start in enumerate(starts):
        stop = start + counts[i]
        for j, value in enumerate(range(start, stop)):
            i_idx[h] = i
            j_idx[h] = j
            values[h] = value
            h += 1
    return (i_idx, j_idx, values)
