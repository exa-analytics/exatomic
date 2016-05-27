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
from exa import Symbolic, DataFrame


lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'k': 7, 'l': 8,
        'm': 9, 'px': 1, 'py': 1, 'pz': 1}
ml_count = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'k': 15,
            'l': 17, 'm': 19}


class BasisSetSummary(DataFrame):
    '''
    Stores a summary of the basis set(s) used in the universe.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | id                | str/cat  | code specific identifier (e.g. tag)       |
    +-------------------+----------+-------------------------------------------+
    | name              | str/cat  | common basis set name/description         |
    +-------------------+----------+-------------------------------------------+
    | function_count    | int      | total number of basis functions           |
    +-------------------+----------+-------------------------------------------+

    Note:
        The function count corresponds to the number of linearly independent
        basis functions as provided by the basis set definition and used within
        the code in solving the quantum mechanical eigenvalue problem (e.g.
        s = 1, p = 3, d = 5, etc.).
    '''
    _columns = ['id', 'name', 'function_count']
    _indices = ['basis_set']
    _categories = {'name': str, 'id': str}


class BasisSet(DataFrame):
    '''
    Base class for description of a basis set. Stores the parameters of the
    individual (sometimes called primitive) functions used in the basis.
    '''
    pass


class SlaterBasisSet(BasisSet):
    '''
    Stores information about a Slater type basis set.

    .. math::

        r = \\left(\\left(x - A_{x}\\right)^{2} + \\left(x - A_{y}\\right)^{2} + \\left(z - A_{z}\\right)^{2}\\right)^{\\frac{1}{2}} \\\\
        f\\left(x, y, z\\right) = \\left(x - A_{x}\\right)^{i}\\left(x - A_{y}\\right)^{j}\left(z - A_{z}\\right)^{k}r^{m}e^{-\\alpha r}
    '''
    pass


class GaussianBasisSet(BasisSet):
    '''
    Stores information about a Gaussian type basis set.

    A Gaussian type basis set is described by primitive Gaussian functions :math:`f\\left(x, y, z\\right)`
    of the form:

    .. math::

        r^{2} = \\left(x - A_{x}\\right)^{2} + \\left(x - A_{y}\\right)^{2} + \\left(z - A_{z}\\right)^{2} \\\\
        f\\left(x, y, z\\right) = \\left(x - A_{x}\\right)^{l}\\left(x - A_{y}\\right)^{m}\\left(z - A_{z}\\right)^{n}e^{-\\alpha r^{2}}

    Note that :math:`l`, :math:`m`, and :math:`n` are not quantum numbers but positive integers
    (including zero) whose sum defines the orbital angular momentum of the primitive function.
    Each primitive function is centered on a given atom with coordinates :math:`\\left(A_{x}, A_{y}, A_{z}\\right)`.
    A basis function in this basis set is a sum of one or more primitive functions:

    .. math::

        g_{i}\\left(x, y, z\\right) = \\sum_{j=1}^{N_{i}}c_{ij}f_{ij}\\left(x, y, z\\right)

    Each primitive function :math:`f_{ij}` is parameterically dependent on its associated atom's
    nuclear coordinates and specific values of :math:`\\alpha`, :math:`l`, :math:`m`, and :math:`n`.
    For convenience in data storage, each primitive function record contains its value of
    :math:`\\alpha` and coefficient (typically called the contraction coefficient) :math:`c`.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | alpha             | float    | value of :math:`\\alpha`                  |
    +-------------------+----------+-------------------------------------------+
    | c                 | float    | value of the contraction coefficient      |
    +-------------------+----------+-------------------------------------------+
    | basis_function    | int/cat  | basis function group identifier           |
    +-------------------+----------+-------------------------------------------+
    | shell             | str/cat  | chemists' notation orbital shell          |
    +-------------------+----------+-------------------------------------------+
    | basis_set         | int/cat  | basis set identifier                      |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['alpha', 'c', 'basis_function', 'shell', 'basis_set']
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
                                            lambda g: (g['shell'].map(ml_count)).values[0]).sum())


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
