# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
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
import pandas as pd
import numpy as np
from exa import DataFrame

from exatomic.algorithms.basis import (lmap, spher_ml_count, enum_cartesian,
                                       gaussian_cartesian, rlmap,
                                       cart_lml_count, spher_lml_count,
                                       _vec_normalize, _wrap_overlap, lorder,
                                       _vec_sto_normalize, _ovl_indices,
                                       solid_harmonics, car2sph)

# Abbreviations
# NCartPrim -- Total number of cartesian primitive functions
# NSphrPrim -- Total number of spherical primitive functions
# NPrim     -- Total number of primitive basis functions (one of the above)
# Nbas      -- Total number of contracted basis functions
# NbasTri   -- Nbas * (Nbas + 1) // 2


# Truncated NPrim dimensions indexed to save space
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
    _categories = {'L': np.int64, 'set': np.int64, 'frame': np.int64}

    @property
    def lmax(self):
        return self['L'].cat.as_ordered().max()

    @property
    def shells(self):
        return [lorder[l] for l in self.L.unique()]

    @property
    def nshells(self):
        return len(self.shells)

    def _sets(self):
        """Group by basis set."""
        return self.groupby('set')

    def functions_by_shell(self):
        """Return a series of (l, n function) pairs per set."""
        mi = self._sets().apply(
            lambda x: x.groupby('shell').apply(
            lambda y: y['L'].values[0]).value_counts())
        if type(mi) == pd.DataFrame:
            return pd.Series(mi.values[0], index=pd.MultiIndex.from_product(
                             [mi.index.values, mi.columns.values],
                             names=['set', 'L']))
        mi.index.names = ['set', 'L']
        return mi.sort_index()

    def primitives_by_shell(self):
        """Return a series of (l, n primitive) pairs per set."""
        return self._sets_ls().apply(
            lambda y: y.apply(
            lambda z: len(z['alpha'].unique()))).T.unstack()

    def primitives(self, lml_count):
        """Total number of primitive functions per set."""
        return self._sets().apply(
            lambda x: x.groupby('alpha').apply(
                lambda y: y.groupby('L').apply(
                    lambda z: z.iloc[0])['L'].map(lml_count).sum()).sum())

    def __init__(self, *args, spherical=True, gaussian=True, **kwargs):
        super(BasisSet, self).__init__(*args, **kwargs)
        self.spherical = spherical
        self.gaussian = gaussian
        norm = _vec_normalize if gaussian else _vec_sto_normalize
        colm = 'L' if gaussian else 'n'
        self['N'] = norm(self['alpha'].values, self[colm].values)
        self['Nd'] = self['d'] * self['N']



# Nbas dimensions
class BasisSetOrder(DataFrame):
    """
    BasisSetOrder uniquely determines the basis function ordering scheme for
    a given :class:`~exatomic.universe.Universe`. This table is provided to
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


# More general than the Overlap matrix but
# has NBasTri dimensions
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

    def square(self, frame=0):
        nbas = np.round(np.roots([1, 1, -2 * self.shape[0]])[1]).astype(np.int64)
        tri = self[self['frame'] == frame].pivot('chi0', 'chi1', 'coef').fillna(value=0)
        return tri + tri.T - np.eye(nbas)

    @classmethod
    def from_column(cls, source):
        """Create an Overlap from a file with just the array of coefficients or
        an array of the values directly."""
        # Assuming source is a file of triangular elements of the overlap matrix
        try: vals = pd.read_csv(source, header=None).values.flatten()
        except: vals = source
        # Reverse engineer the number of basis functions given len(ovl) = n * (n + 1) / 2
        nbas = np.round(np.roots((1, 1, -2 * vals.shape[0]))[1]).astype(np.int64)
        # Index chi0 and chi1, they are interchangeable as overlap is symmetric
        chis = _ovl_indices(nbas, vals.shape[0])
        return cls(pd.DataFrame.from_dict({'chi0': chis[:, 0],
                                           'chi1': chis[:, 1],
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


# NPrim dimensions
# Additionally, from_universe returns additional matrices with
# (NCartPrim, NSphrPrim) dimensions and (NPrim, NBas) dimensions
class Primitive(DataFrame):
    """
    Notice: Primitive is just a join of basis set and atom, re-work needed.
    Contains the required information to perform molecular integrals. Some
    repetition of data with GaussianBasisSet but for convenience also produced
    here. This is an intermediary DataFrame which won't exist in a production
    implementation

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | xa                | float    | center in x direction of primitive        |
    +-------------------+----------+-------------------------------------------+
    | ya                | float    | center in y direction of primitive        |
    +-------------------+----------+-------------------------------------------+
    | za                | float    | center in z direction of primitive        |
    +-------------------+----------+-------------------------------------------+
    | alpha             | float    | value of :math:`\\alpha`, the exponent     |
    +-------------------+----------+-------------------------------------------+
    | N                 | float    | value of the normalization constant       |
    +-------------------+----------+-------------------------------------------+
    | l                 | int      | pre-exponential power of x                |
    +-------------------+----------+-------------------------------------------+
    | m                 | int      | pre-exponential power of y                |
    +-------------------+----------+-------------------------------------------+
    | n                 | int      | pre-exponential power of z                |
    +-------------------+----------+-------------------------------------------+
    | L                 | int/cat  | sum of l + m + n                          |
    +-------------------+----------+-------------------------------------------+
    | set               | int/cat  | unique basis set identifier               |
    +-------------------+----------+-------------------------------------------+
    """
    _columns = ['xa', 'ya', 'za', 'alpha', 'N', 'l', 'm', 'n', 'L', 'set']
    _index = 'primitive'
    _categories = {'l': np.int64, 'm': np.int64, 'n': np.int64, 'L': np.int64}

    def primitive_overlap(self):
        """Compute the complete primitive cartesian overlap matrix."""
        cols = ['xa', 'ya', 'za', 'l', 'm', 'n', 'N', 'alpha']
        self._revert_categories()
        chi0, chi1, ovl = _wrap_overlap(*(self[col].values for col in cols))
        self._set_categories()
        return Overlap.from_dict({'chi0': chi0, 'chi1': chi1,
                                  'coef': ovl, 'frame': 0})

    @classmethod
    def from_universe(cls, uni, grpby='L', frame=None, debug=True):
        """
        Generate the DF and associated contraction matrices. Currently
        spits out the Primitive dataframe along with cartesian to spherical
        and contraction matrices, as this class will disappear in an appropriate
        implementation.

        Args
            uni (exatomic.container.Universe): a universe with basis set
            grpby (str): one of 'L' or 'shell' for different basis sets
            frame (int): always blue?
        """
        frame = uni.atom.nframes - 1 if frame is None else frame
        uni.basis_set._set_categories()
        sh = solid_harmonics(uni.basis_set.lmax)
        uni.basis_set._revert_categories()
        sets = uni.basis_set.cardinal_groupby().get_group(frame).groupby('set')
        funcs = uni.basis_set_order.cardinal_groupby().get_group(frame).groupby('center')
        atom = uni.atom.cardinal_groupby().get_group(frame)
        cart = gaussian_cartesian if uni.meta['program'] == 'gaussian' else enum_cartesian
        conv = car2sph(sh, cart)

        cprim = uni.atom.set.map(uni.basis_set.primitives(cart_lml_count)).sum()
        sprim = uni.atom.set.map(uni.basis_set.primitives(spher_lml_count)).sum()
        ncont = len(uni.basis_set_order.index)
        if uni.basis_set.spherical:
            contdim = sprim
            lml_count = spher_lml_count
        else:
            contdim = cprim
            lml_count = cart_lml_count

        cols = ['xa', 'ya', 'za', 'alpha', 'N', 'l', 'm', 'n', 'L', 'set']
        typs = ['f8', 'f8', 'f8', 'f8', 'f8', 'i8', 'i8', 'i8', 'i8', 'i8']
        primdf = np.empty((cprim,), dtype=[(i, j) for i, j in zip(cols, typs)])
        sphrdf = np.zeros((cprim, sprim), dtype=np.float64)
        contdf = np.zeros((contdim, ncont), dtype=np.float64)

        if debug:
            print('Overlap grouping by', grpby)
            print('{} cprims, {} sprims, {} ncont'.format(cprim, sprim, ncont))

        pcnt, ridx, cidx, pidx, sidx = 0, 0, 0, 0, 0
        for i, (seht, x, y, z) in enumerate(zip(atom['set'], atom['x'],
                                                atom['y'], atom['z'])):
            setdf = sets.get_group(seht).groupby(grpby)
            aobas = funcs.get_group(i).groupby(grpby)
            for idx, contsh in aobas:
                if not len(contsh.index): continue
                try: sh = setdf.get_group(idx)
                except: continue
                L = idx if grpby == 'L' else sh['L'].values[0]
                chnk = sh.pivot('alpha', 'shell', 'd').loc[sh.alpha.unique()].fillna(0.0)
                sh = sh.drop_duplicates('alpha')
                pdim, cdim = chnk.shape
                # Minimum primitive information
                for l, m, n in cart[L]:
                    for alpha, N in zip(sh.alpha, sh.N):
                        primdf[pcnt] = (x, y, z, alpha, N, l, m, n, L, seht)
                        pcnt += 1
                # Cartesian to spherical prim
                c2s = conv[L]
                cplus, splus = c2s.shape
                for j in range(pdim):
                    sphrdf[pidx:pidx + cplus,sidx:sidx + splus] = c2s
                    pidx += cplus
                    sidx += splus
                # Spherical primitive to contracted
                for k in range(lml_count[L]):
                    contdf[ridx:ridx + pdim,cidx:cidx + cdim] = chnk.values
                    cidx += cdim
                    ridx += pdim
        return cls(primdf, columns=cols), pd.DataFrame(sphrdf), pd.DataFrame(contdf)
