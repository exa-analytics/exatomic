# -*- coding: utf-8 -*-
'''
Basis Function Manipulation
===============================
Functions for managing and manipulating basis set data.
'''
from exa import _conf


lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'k': 7, 'l': 8,
        'm': 9, 'px': 1, 'py': 1, 'pz': 1}
ml_count = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'k': 15,
            'l': 17, 'm': 19}


def _cartesian_ordering_function(l):
    '''
    Generic function for generating (linearly dependent) sets of cartesian
    Gaussian type functions.

    Args:
        l (int): Orbital angular momentum

    Returns:
        array: Array of powers of x, y, z for cartesian Gaussian type functions

    Note:
        This returns the linearly dependent indices (array) in arbitrary
        order.
    '''
    m = l + 1
    n = (m + 1) * m // 2
    values = np.empty((n, 3), dtype=np.int64)
    h = 0
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if i + j + k == l:
                    values[h] = [i, j, k]
                    h += 1
    return values


if _conf['pkg_numba']:
    from numba import jit
    _cartesian_ordering_function = jit(nopython=True, cache=True)(_cartesian_ordering_function)
