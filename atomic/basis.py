# -*- coding: utf-8 -*-
'''
Basis Set Representations
=============================
Class that support representations of various basis sets.

See Also:
    :mod:`~atomic.algorithms.basis`
'''
import pandas as pd
import numpy as np
from collections import OrderedDict
from exa.numerical import DataFrame


class Basis(DataFrame):
    '''
    Base class for storing basis set data.
    '''
    def per_symbol_counts(self):
        '''
        Compute the number of basis functions per atomic symbolic.

        Returns:
            counts (:class:`~pandas.Series`): Per symbol/symbol+label basis sets.
        '''


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
