# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Basis Set Representations
##############################
This module provides classes that support representations of various basis sets.
There are a handful of basis sets in computational chemistry, the most common of
which are Gaussian type functions, Slater type functions, and plane waves. The
classes provided by this module support not only storage of basis set data, but
also analytical and discrete manipulations of the basis set.

See Also:
    For symbolic and discrete manipulations see :mod:`~exatomic.algorithms.basis`.
'''
import pandas as pd
import numpy as np
from collections import OrderedDict
from exa import DataFrame
#from exatomic.algorithms.basis import spher_ml_count, cart_ml_count


lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'k': 7, 'l': 8,
        'm': 9, 'px': 1, 'py': 1, 'pz': 1}
spher_ml_count = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'k': 15,
                  'l': 17, 'm': 19}


class BasisSet(DataFrame):
    '''
    Description of the basis set name, number of functions, and function types.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | tag               | str/cat  | code specific basis set identifier        |
    +-------------------+----------+-------------------------------------------+
    | name              | str/cat  | common basis set name/description         |
    +-------------------+----------+-------------------------------------------+
    | function_count    | int      | total number of basis functions           |
    +-------------------+----------+-------------------------------------------+
    | symbol            | str/cat  | unique atomic label                       |
    +-------------------+----------+-------------------------------------------+
    | prim_per_atom     | int      | primitive functions per atom              |
    +-------------------+----------+-------------------------------------------+
    | func_per_atom     | int      | basis functions per atom                  |
    +-------------------+----------+-------------------------------------------+
    | primitive_count   | int      | total primitive functions                 |
    +-------------------+----------+-------------------------------------------+
    | function_count    | int      | total basis functions                     |
    +-------------------+----------+-------------------------------------------+

    Note:
        The function count corresponds to the number of linearly independent
        basis functions as provided by the basis set definition and used within
        the code in solving the quantum mechanical eigenvalue problem.
    '''
    _columns = ['tag', 'name', 'function_count']
    _indices = ['set']

        #(e.g.
        #s = 1, p = 3, d = 5, etc.).

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

    Other useful columns can be added to increase compatibility with other functionality.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | index             | int/cat  | basis set identifier                      |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['alpha', 'd', 'basis_function', 'shell']
    _indices = ['primitive']
    _groupbys = ['basis_function']
    _categories = {'basis_set': np.int64, 'shell': str, 'name': str, 'basis_function': np.int64}

    def basis_count(self):
        '''
        Number of basis functions (:math:`g_{i}`) per symbol or label type.

        Returns:
            counts (:class:`~pandas.Series`)
        '''
        return self.groupby('symbol').apply(lambda g: g.groupby('function').apply(
                                            lambda g: (g['shell'].map(spher_ml_count)).values[0]).sum())

class BasisSetOrder(BasisSet):
    '''
    BasisSetOrder uniquely determines the basis function ordering scheme for
    a given :class:`~exatomic.universe.Universe`. shell_function is used instead
    of basis_function in the following table to emphasize that it includes the
    degeneracy from the quantum number :math:`m_{l}`, which may change. This
    table should be used if the ordering scheme cannot be determined
    programmatically.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | shell_function    | int      | shell function index                      |
    +-------------------+----------+-------------------------------------------+
    | symbol            | str      | symbolic atomic center                    |
    +-------------------+----------+-------------------------------------------+
    | center            | int      | numeric atomic center (1-based)           |
    +-------------------+----------+-------------------------------------------+
    | type              | str      | identifier equivalent to (l, ml)          |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['shell_function', 'symbol', 'center', 'type']
    _indices = ['order']
    _categories = {'symbol': str, 'center': np.int64, 'type': str, 'basis_function': np.int64}

class BasisSetMap(BasisSet):
    '''
    BasisSetMap provides the auxiliary information about relational mapping
    between the complete uncontracted primitive basis set and the resultant
    contracted basis set within an :class:`~exatomic.universe.Universe`.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | symbol            | str      | symbolic atomic center                    |
    +-------------------+----------+-------------------------------------------+
    | shell             | str      | string of quantum number l                |
    +-------------------+----------+-------------------------------------------+
    | nprim             | int      | number of primitives within shell         |
    +-------------------+----------+-------------------------------------------+
    | nbasis            | int      | number of basis functions within shell    |
    +-------------------+----------+-------------------------------------------+
    | cartesian         | bool     | shell is cartesian                        |
    +-------------------+----------+-------------------------------------------+
    | spherical         | bool     | shell is spherical                        |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['symbol', 'shell', 'nprim', 'nbasis', 'cartesian', 'spherical']
    _indices = ['index']
    _categories = {'symbol': str, 'shell': str, 'nprim': np.int64,
                   'nbasis': np.int64, 'cartesian': bool, 'spherical': bool}


class Overlap(DataFrame):
    '''
    Overlap enumerates the overlap matrix elements between basis functions in
    a contracted basis set. Currently nothing disambiguates between the
    primitive overlap matrix and the contracted overlap matrix. As it is
    square symmetric, only n_basis_functions * (n_basis_functions + 1) / 2
    rows are stored.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | chi1              | int      | first basis function                      |
    +-------------------+----------+-------------------------------------------+
    | chi2              | int      | second basis function                     |
    +-------------------+----------+-------------------------------------------+
    | coefficient       | float    | overlap matrix element                    |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['chi1', 'chi2', 'coefficient']
    _indices = ['index']

    def square(self):
        nbas = np.floor(np.sqrt(self.shape[0] * 2))
        return self.pivot('chi1', 'chi2', 'coefficient').fillna(value=0) + \
               self.pivot('chi2', 'chi1', 'coefficient').fillna(value=0) - np.eye(nbas)


class PlanewaveBasisSet(BasisSet):
    '''
    '''
    pass



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
