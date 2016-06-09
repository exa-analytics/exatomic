# -*- coding: utf-8 -*-
'''
Basis Set Representations
=============================
This module provides classes that support representations of various basis sets.
There are a handful of basis sets in computational chemistry, the most common of
which are Gaussian type functions, Slater type functions, and plane waves. The
classes provided by this module support not only storage of basis set data, but
also analytical and discrete manipulations of the basis set.

See Also:
    For symbolic and discrete manipulations see :mod:`~atomic.algorithms.basis`.
'''
import pandas as pd
import numpy as np
from collections import OrderedDict
from exa import DataFrame


lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'k': 7, 'l': 8,
        'm': 9, 'px': 1, 'py': 1, 'pz': 1}
ml_count = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'k': 15,
            'l': 17, 'm': 19}


class BasisSet(DataFrame):
    '''
    Description of the basis set name, number of functions, and function types.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | tag                | str/cat  | code specific identifier (e.g. tag)      |
    +-------------------+----------+-------------------------------------------+
    | name              | str/cat  | common basis set name/description         |
    +-------------------+----------+-------------------------------------------+
    | function_count    | int      | total number of basis functions           |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['tag', 'name', 'function_count']
    _indices = ['set']


class BasisFunction(DataFrame):
    '''
    Definition of individual basis functions used in a given universe.
    '''
    pass


class GaussianBasis(BasisFunction):
    '''
    Description of primitive functions used to construct a (contracted)
    Gaussian basis set.

    A real contracted Gaussian basis function, in cartesian space, is defined:

    .. math::

        \\Chi_{j} = \\sum_{k=1}^{K}d_{jk}NPe^{-\\alpha r^{2}}

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | alpha             | float    | value of :math:`\\alpha`                  |
    +-------------------+----------+-------------------------------------------+
    | d                 | float    | contraction coefficient                   |
    +-------------------+----------+-------------------------------------------+
    | j                 | int/cat  | basis function identifier                 |
    +-------------------+----------+-------------------------------------------+
    | l                 | str/cat  | orbital angular momentum                  |
    +-------------------+----------+-------------------------------------------+
    | set               | int/cat  | basis set index                           |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['alpha', 'd', 'j', 'l', 'set']
    _indices = ['index']
    _groupbys = ['set', 'j']


class CartesianGTFOrder(DataFrame):
    '''
    Stores cartesian basis function order with respect to basis function label.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | x                 | int      | power of x                                |
    +-------------------+----------+-------------------------------------------+
    | y                 | int      | power of y                                |
    +-------------------+----------+-------------------------------------------+
    | z                 | int      | power of z                                |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['x', 'y', 'z']
    _indices = ['cart_order']

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
    _indices = ['spherical_order']

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
