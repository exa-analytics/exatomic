# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Basis Set Representations
##############################
This module provides classes that support representations of various basis sets.
There are a handful of basis sets in computational chemistry, the most common of
which are Gaussian type functions, Slater type functions, and plane waves. The
classes provided by this module support not only storage of basis set data, but
also analytical and discrete manipulations of the basis set.

See Also:
    For symbolic and discrete manipulations see :mod:`~exatomic.algorithms.basis`.
"""
import os
import numpy as np
import pandas as pd
from io import StringIO

from exa import DataFrame
from exatomic.algorithms.basis import cart_lml_count, spher_lml_count
from exatomic.algorithms.numerical import _tri_indices, _square, Shell


class BasisSet(DataFrame):
    """
    Stores information about a basis set. Common basis set types in use for
    electronic structure calculations usually consist of Gaussians or Slater
    Type Orbitals (STOs). Both types usually employ atom-centered basis functions,
    where each basis function resides on a given atom with coordinates
    :math:`\\left(A_{x}, A_{y}, A_{z}\\right)`. For Gaussian basis sets, the
    functional form of :math:`f\\left(x, y, z\\right)` is:

    .. math::

        r^{2} = \\left(x - A_{x}\\right)^{2} + \\left(x - A_{y}\\right)^{2} + \\left(z - A_{z}\\right)^{2} \\\\
        f\\left(x, y, z\\right) = \\left(x - A_{x}\\right)^{l}\\left(x - A_{y}\\right)^{m}\\left(z - A_{z}\\right)^{n}e^{-\\alpha r^{2}}

    where :math:`l`, :math:`m`, and :math:`n` are not quantum numbers but positive
    integers (including zero) whose sum defines the orbital angular momentum of
    each function and :math:`alpha` governs the exponential decay of the given
    function. Gaussian basis functions are usually constructed from multiple
    primitive Gaussians, with fixed contraction coefficients. Therefore, a basis
    function consists of the sum of one or more primitive functions:

    .. math::

        g_{i}\\left(x, y, z\\right) = \\sum_{j=1}^{N_{i}}c_{ij}f_{ij}\\left(x, y, z\\right)

    Alternatively, STOs are usually not constructed from linear combinations
    of multiple primitives, and differ from Gaussian type functions in that they
    do not contain an exponent in the :math:`r` term of the exponential decay.
    These functions have 2 main benefits; an adequate description of the cusp
    in the density at the nucleus, and the appropriate long-range decay behavior.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | alpha             | float    | exponent                                  |
    +-------------------+----------+-------------------------------------------+
    | shell             | int      | group of primitives                       |
    +-------------------+----------+-------------------------------------------+
    | set               | int/cat  | unique basis set identifier               |
    +-------------------+----------+-------------------------------------------+
    | d                 | float    | contraction coefficient                   |
    +-------------------+----------+-------------------------------------------+
    | L                 | int      | orbital angular momentum                  |
    +-------------------+----------+-------------------------------------------+
    """
    _columns = ['alpha', 'd', 'shell', 'L', 'set']
    _cardinal = ('frame', np.int64)
    _index = 'function'
    _categories = {'L': np.int64, 'set': np.int64, 'frame': np.int64, 'norm': str}

    @property
    def _constructor(self):
        return BasisSet

    @property
    def lmax(self):
        return self['L'].cat.as_ordered().max()

    def shells(self):
        def _shell_gau(df):
            piv = ('alpha', 'shell', 'd')
            alphas = df.alpha.unique()
            piv = df.pivot(*piv).loc[alphas].fillna(0.)
            nprim, ncont = piv.shape
            return Shell(piv.values.flatten(), alphas, nprim, ncont, df.L.values[0],
                         df.norm.values[0], self.gaussian, None, None)
        def _shell_sto(df):
            piv = ('alpha', 'shell', 'd')
            alphas = df.alpha.unique()
            piv = df.pivot(*piv).loc[alphas].fillna(0.)
            nprim, ncont = piv.shape
            return Shell(piv.values.flatten(), alphas, nprim, ncont, df.L.values[0],
                        self.spherical, self.gaussian, df.r.values, df.n.values)
        if self.gaussian:
            if 'norm' not in self.columns: self.spherical_by_shell()
            return self.groupby(['set', 'L']).apply(_shell_gau).reset_index()
        return self.groupby(['set', 'L']).apply(_shell_sto).reset_index()

    def spherical_by_shell(self):
        """Pre-alpha."""
        self['L'] = self['L'].astype(np.int64)
        self['norm'] = self['L'].apply(lambda L: L > 1)
        self['L'] = self['L'].astype('category')

    def functions_by_shell(self):
        """Return a series of n functions per (set, L).
        This does not include degenerate functions."""
        return self.groupby(['set', 'L'])['shell'].nunique()

    def primitives_by_shell(self):
        """Return a series of n primitives per (set, L).
        This does not include degenerate primitives."""
        return self.groupby(['set', 'L'])['alpha'].nunique()

    def functions(self, spherical):
        """Return a series of n functions per (set, L).
        This does include degenerate functions."""
        if spherical:
            mapper = lambda x: spher_lml_count[x]
        else:
            mapper = lambda x: cart_lml_count[x]
        n = self.functions_by_shell()
        return n * n.index.get_level_values('L').map(mapper)

    def primitives(self, spherical):
        """Return a series of n primitives per (set, L).
        This does include degenerate primitives."""
        if spherical:
            mapper = lambda x: spher_lml_count[x]
        else:
            mapper = lambda x: cart_lml_count[x]
        n = self.primitives_by_shell()
        return n * n.index.get_level_values('L').map(mapper)

    def __init__(self, *args, **kwargs):
        spherical = kwargs.pop("spherical", True)
        gaussian = kwargs.pop("gaussian", True)
        super(BasisSet, self).__init__(*args, **kwargs)
        self.spherical = spherical
        self.gaussian = gaussian


class BasisSetOrder(DataFrame):
    """
    BasisSetOrder uniquely determines the basis function ordering scheme for
    a given :class:`~exatomic.core.universe.Universe`. This table is provided to
    make transparent the characteristic ordering scheme of various quantum
    codes. Either (L, ml) or (l, m, n) must be provided to have access to
    orbital visualization functionality.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | center            | int      | atomic center                             |
    +-------------------+----------+-------------------------------------------+
    | L                 | int      | orbital angular momentum                  |
    +-------------------+----------+-------------------------------------------+
    | shell             | int      | group of primitives                       |
    +-------------------+----------+-------------------------------------------+
    | ml                | int      | magnetic quantum number                   |
    +-------------------+----------+-------------------------------------------+
    | l                 | int      | power in x                                |
    +-------------------+----------+-------------------------------------------+
    | m                 | int      | power in y                                |
    +-------------------+----------+-------------------------------------------+
    | n                 | int      | power in z                                |
    +-------------------+----------+-------------------------------------------+
    | r                 | int      | power in r (optional - for STOs)          |
    +-------------------+----------+-------------------------------------------+
    | prefac            | float    | prefactor (optional - for STOs)           |
    +-------------------+----------+-------------------------------------------+
    """
    _columns = ['center', 'L', 'shell']
    _index = 'chi'
    _cardinal = ('frame', np.int64)
    _categories = {'L': np.int64}

    @property
    def _constructor(self):
        return BasisSetOrder


class Overlap(DataFrame):
    """
    Overlap enumerates the overlap matrix elements between basis functions in
    a contracted basis set. Currently nothing disambiguates between the
    primitive overlap matrix and the contracted overlap matrix. As it is
    square symmetric, only n_basis_functions * (n_basis_functions + 1) / 2
    rows are stored.

    See Gramian matrix for more on the general properties of the overlap matrix.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | frame             | int/cat  | non-unique integer                        |
    +-------------------+----------+-------------------------------------------+
    | chi0              | int      | first basis function                      |
    +-------------------+----------+-------------------------------------------+
    | chi1              | int      | second basis function                     |
    +-------------------+----------+-------------------------------------------+
    | coef              | float    | overlap matrix element                    |
    +-------------------+----------+-------------------------------------------+
    """
    _columns = ['chi0', 'chi1', 'coef', 'frame']
    _index = 'index'

    @property
    def _constructor(self):
        return Overlap

    def square(self, frame=0, column='coef'):
        """Return a 'square' matrix DataFrame of the Overlap."""
        sq = pd.DataFrame(_square(self[column].values))
        sq.index.name = 'chi0'
        sq.columns.name = 'chi1'
        return sq

    @classmethod
    def from_column(cls, source):
        """Create an Overlap from a file with just the array of coefficients or
        an array of the values directly."""
        # Assuming source is a file of triangular elements of the overlap matrix
        if isinstance(source, np.ndarray):
            vals = source
        elif isinstance(source, str):
            if os.sep in source:
                vals = pd.read_csv(source, header=None).values.flatten()
            else:
            # except FileNotFoundError:
                vals = pd.read_csv(StringIO(source), header=None).values.flatten()
        chi0, chi1 = _tri_indices(vals)
        return cls(pd.DataFrame.from_dict({'chi0': chi0,
                                           'chi1': chi1,
                                           'coef': vals,
                                           'frame': 0}))

    @classmethod
    def from_square(cls, df):
        ndim = df.shape[0]
        try: arr = df.values
        except: arr = df
        arlen = ndim * (ndim + 1) // 2
        ret = np.empty((arlen,), dtype=[('chi0', 'i8'),
                                        ('chi1', 'i8'),
                                        ('coef', 'f8'),
                                        ('frame', 'i8')])
        cnt = 0
        for i in range(ndim):
            for j in range(i + 1):
                ret[cnt] = (i, j, arr[i, j], 0)
                cnt += 1
        return cls(ret)
