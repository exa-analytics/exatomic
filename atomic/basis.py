# -*- coding: utf-8 -*-
'''
Basis Functions
=============================
Representations of commonly used basis functions
'''
import pandas as pd
import numpy as np
import sympy as sy
from collections import OrderedDict
from exa import _conf
from exa.analytical import Symbolic
from exa.numerical import DataFrame


lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'k': 7, 'l': 8,
        'm': 9, 'px': 1, 'py': 1, 'pz': 1}
ml_count = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'k': 15,
            'l': 17, 'm': 19}


class Basis(DataFrame):
    '''
    Base class for storing basis set data.
    '''
    pass


class GTFBasis(Basis):
    '''
    Stores information about a Gaussian type basis set.
    '''
    _columns = ['alpha', 'c', 'function', 'shell', 'symbol']
    _indices = ['basis']
    _categories = {'symbol': str, 'shell': str, 'name': str}


class CartesianGTFOrder(DataFrame):
    '''
    Stores cartesian basis function order with respect to basis function label.
    '''
    _columns = ['x', 'y', 'z']
    _indices = ['order']

    @classmethod
    def from_lmax_order(cls, lmax, ordering_function):
        '''
        Generate the dataframe of cartesian basis function ordering with
        respect to spin angular momentum.

        Args:
            lmax (int): Maximum value of orbital angular momentum
            ordering_function: Cartesian ordering function (code specific)
        '''
        df = pd.DataFrame(np.concatenate([ordering_function(l) for l in range(lmax + 1)]),
                          columns=['x', 'y', 'z'])
        return cls(df)

    def symbolic_keys(self):
        '''
        Generate the enumerated symbolic keys (e.g. 'x', 'xx', 'xxyy', etc.)
        associated with each row for ordering purposes.
        '''
        x = self['x'].apply(lambda i: 'x' * i).astype(str)
        y = self['y'].apply(lambda i: 'y' * i).astype(str)
        z = self['z'].apply(lambda i: 'z' * i).astype(str)
        return x + y + z


class SphericalGTFOrder(DataFrame):
    '''
    Stores order of spherical basis functions with respect to angular momenta.
    '''
    _columns = ['l', 'ml']
    _indices = ['order']

    @classmethod
    def from_lmax_order(cls, lmax, ordering_function):
        '''
        Generate the spherical basis function ordering with respect
        to spin angular momentum.

        Args:
            lmax (int): Maximum value of orbital angular momentum
            ordering_function: Spherical ordering function (code specific)
        '''
        data = OrderedDict([(l, ordering_function(l)) for l in range(lmax + 1)])
        l = [k for k, v in data.items() for i in range(len(v))]
        ml = np.concatenate(list(data.values()))
        df = pd.DataFrame.from_dict({'l': l, 'ml': ml})
        return cls(df)

    def symbolic_keys(self, l=None):
        '''
        Generate the enumerated symbolic keys (e.g. '(0, 0)', '(1, -1)', '(2, 2)',
        etc.) associated with each row for ordering purposes.
        '''
        obj = zip(self['l'], self['ml'])
        if l is None:
            return list(obj)
        return [kv for kv in obj if kv[0] == l]


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
