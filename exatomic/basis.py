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
from traitlets import Float, Int, Dict, Unicode
from exa import DataFrame

from exatomic.algorithms.basis import (lmap, spher_ml_count, enum_cartesian,
                                       cart_lml_count, spher_lml_count,
                                       _vec_normalize, _wrap_overlap,
                                       solid_harmonics, car2sph_transform_matrices)


class TestBasis(DataFrame):
    _columns = ['alpha_dict', 'd_dict', 'l_dict']
    _traits = ['alpha_dict', 'd_dict', 'l_dict']
    _indices = ['set']


class BasisSetSummary(DataFrame):
    '''
    Stores a summary of the basis set(s) used in the universe.

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
    | prim_X            | int      | X = shell primitive functions             |
    +-------------------+----------+-------------------------------------------+
    | bas_X             | int      | X = shell basis functions                 |
    +-------------------+----------+-------------------------------------------+
    | frame             | int/cat  | non-unique integer                        |
    +-------------------+----------+-------------------------------------------+

    Note:
        The function count corresponds to the number of linearly independent
        basis functions as provided by the basis set definition and used within
        the code in solving the quantum mechanical eigenvalue problem.
    '''
    _columns = ['tag', 'name', 'func_per_atom']
    _indices = ['set']
    _groupbys = ['frame']
    _categories = {'tag': str}


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

    Each primitive function :math:`f_{ij}` is parametrically dependent on its associated atom's
    nuclear coordinates and specific values of :math:`\\alpha`, :math:`l`, :math:`m`, and :math:`n`.
    For convenience in data storage, each primitive function record contains its value of
    :math:`\\alpha` and coefficient (typically called the contraction coefficient) :math:`c`.
    shell_function does not include degeneracy due to :math:`m_{l}` but separates exponents
    and coefficients that have the same angular momentum values.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | alpha             | float    | value of :math:`\\alpha`                   |
    +-------------------+----------+-------------------------------------------+
    | d                 | float    | value of the contraction coefficient      |
    +-------------------+----------+-------------------------------------------+
    | shell_function    | int/cat  | basis function group identifier           |
    +-------------------+----------+-------------------------------------------+
    | L                 | int/cat  | orbital angular momentum quantum number   |
    +-------------------+----------+-------------------------------------------+
    | set               | int/cat  | index of unique basis set per unique atom |
    +-------------------+----------+-------------------------------------------+
    | frame             | int/cat  | non-unique integer                        |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['alpha', 'd', 'shell_function', 'L', 'set']
    _indices = ['function']
    _traits = ['shell_function']
    #_groupbys = ['frame']
    #_precision = {'alpha': 8, 'd': 8}
    _categories = {'set': np.int64, 'L': np.int64, 'shell_function': np.int64,
                   'frame': np.int64}

    def _custom_traits(self):
        alphas = self.groupby('frame').apply(
                 lambda x: x.groupby('set').apply(
                 lambda y: y.groupby('shell_function').apply(
                 lambda z: z['alpha'].values).values)).to_json(orient='values')
        #alphas = Unicode(''.join(['[', alphas, ']'])).tag(sync=True)
        alphas = Unicode(alphas).tag(sync=True)

        ds = self.groupby('frame').apply(
             lambda x: x.groupby('set').apply(
             lambda y: y.groupby('shell_function').apply(
             lambda z: z['d'].values).values)).to_json(orient='values')
        #ds = Unicode(''.join(['[', ds, ']'])).tag(sync=True)
        ds = Unicode(ds).tag(sync=True)

        ls = self.groupby('frame').apply(
             lambda x: x.groupby('set').apply(
             lambda y: y.groupby('shell_function').apply(
             lambda z: z['L'].astype(np.int64).values).values)).to_json(orient='values')
        #ls = Unicode(''.join(['[', ls, ']'])).tag(sync=True)
        ls = Unicode(ls).tag(sync=True)

        return {'gaussianbasisset_d': ds, 'gaussianbasisset_l': ls,
                'gaussianbasisset_alpha': alphas}

    def basis_count(self):
        '''
        Number of basis functions (:math:`g_{i}`) per symbol or label type.

        Returns:
            counts (:class:`~pandas.Series`)
        '''
        return self.groupby('symbol').apply(lambda g: g.groupby('function').apply(
                                            lambda g: (g['shell'].map(spher_ml_count)).values[0]).sum())

    def _check(self):
        if 'N' not in self.columns:
            self._normalize()

#    @classmethod
#    def expand(cls, universe, inplace=False):
#        '''
#        The minimum information specified by a basis set does not include
#        expansion due to degeneracy from m_l. This will expand the basis in a
#        systematic cartesian ordering convention to generate the full cartesian
#        basis. The universe argument must already have a universe with atom,
#        basis_set_summary, and gaussian_basis_set attributes.
#        '''
#        bases = universe.gaussian_basis_set[abs(universe.gaussian_basis_set['d']) > 0].groupby('set')
#        primdf = []
#        shfunc, func = -1, -1
#        for seht, x, y, z in zip(universe.atom['set'], universe.atom['x'],
#                                 universe.atom['y'], universe.atom['z']):
#            summ = universe.basis_set_summary.ix[seht]
#            b = bases.get_group(seht).groupby('shell_function')
#            for sh in range(len(b)):
#                prims = b.get_group(sh)
#                l = prims['L'].cat.as_ordered().max()
#                shfunc += 1
#                for l, m, n in enum_cartesian[l]:
#                    func += 1
#                    for alpha, d in zip(prims['alpha'], prims['d']):
#                        primdf.append([x, y, z, alpha, d, l, m, n, l + m + n, sh, shfunc, func, seht])
#        primdf = pd.DataFrame(primdf)
#        primdf.columns = ['xa', 'ya', 'za', 'alpha', 'd', 'l', 'm', 'n', 'L', 'shell_function', 'shell', 'func', 'set']
#        if inplace:
#            universe.gaussian_basis_set = primdf
#        else:
#            return cls(primdf)


class Primitive(DataFrame):
    _columns = ['xa', 'ya', 'za', 'alpha', 'd', 'l', 'm', 'n', 'L']
    _indices = ['primitive']
    _categories = {'l': np.int64, 'm': np.int64, 'n': np.int64, 'L': np.int64}

    def _normalize(self):
        '''
        Often primitives come unnormalized. This fixes that.
        '''
        self['N'] = _vec_normalize(self['alpha'].values,
                                   self['l'].astype(np.int64).values,
                                   self['m'].astype(np.int64).values,
                                   self['n'].astype(np.int64).values)


    def _cartesian_contraction_matrix(self, l=False):
        '''
        Generates the (nprim,ncont) matrix needed to reduce the
        dimensionality of the primitive basis to the contracted
        cartesian basis.
        '''
        bfns = self.groupby('func')
        contmat = np.zeros((len(self), len(bfns)), dtype=np.float64)
        cnt = 0
        if l:
            l = np.zeros(len(bfns), dtype=np.int64)
            for bfn, cont in bfns:
                ln = len(cont)
                contmat[cnt:cnt + ln, bfn] = cont['d'].values
                l[bfn] = cont['L'].values[0]
                cnt += ln
            return contmat, l
        for bfn, cont in bfns:
            ln = len(cont)
            contmat[cnt:cnt + ln, bfn] = cont['d'].values
            cnt += ln
        return contmat

    def _spherical_contraction_matrix(self):
        '''
        Generates the (nprim,ncont) matrix needed to reduce the
        dimensionality of the primitive basis to the contracted
        spherical basis.
        '''
        pass


    def _spherical_from_cartesian(self):
        '''
        Reduces the dimensionality of the contracted cartesian
        basis to the contracted spherical basis.
        '''
        print('warning: this is not correct')
        lmax = self['L'].cat.as_ordered().max()
        primS = self.primitive_overlap().square()
        cartprim, ls = self._cartesian_contraction_matrix(l=True)
        contracted = pd.DataFrame(np.dot(np.dot(cartprim.T, primS), cartprim))
        sh = solid_harmonics(lmax)
        sphtrans = car2sph_transform_matrices(sh, lmax)
        bfns = self.groupby('func')
        lcounts = bfns.apply(lambda y: y['L'].values[0]).value_counts()
        for l, lc in lcounts.items():
            lcounts[l] = lc * spher_lml_count[l] // cart_lml_count[l]
        lc = lcounts.sum()
        spherical = np.zeros((contracted.shape[0], lc), dtype=np.float64)
        ip = 0
        ic = 0
        while ip < lc:
            l = ls[ic]
            if l < 2:
                spherical[:,ic] = contracted[ic]
                ip += 1
                ic += 1
            else:
                cspan = ic + cart_lml_count[l]
                sspan = ip + spher_lml_count[l]
                carts = contracted[list(range(ic, cspan))]
                trans = np.dot(carts, sphtrans[l].T)
                spherical[:,ip:sspan] = trans
                ip += spher_lml_count[l]
                ic += cart_lml_count[l]
        return pd.DataFrame(np.dot(np.dot(spherical.T, contracted), spherical))


    def primitive_overlap(self):
        '''
        Computes the complete primitive cartesian overlap matrix.
        '''
        if 'N' not in self.columns:
            self._normalize()
        chi1, chi2, overlap =  _wrap_overlap(self['xa'].values,
                                             self['ya'].values,
                                             self['za'].values,
                                             self['l'].astype(np.int64).values,
                                             self['m'].astype(np.int64).values,
                                             self['n'].astype(np.int64).values,
                                             self['N'].values, self['alpha'].values)
        return Overlap.from_dict({'chi1': chi1, 'chi2': chi2,
                                  'coefficient': overlap,
                                  'frame': [0] * len(chi1)})


    def contracted_cartesian_overlap(self):
        primS = self.primitive_overlap().square()
        contprim = self._cartesian_contraction_matrix()
        square = pd.DataFrame(np.dot(np.dot(contprim.T, primS), contprim))
        return Overlap.from_square(square)

    def contracted_spherical_overlap(self):
        return self._spherical_from_cartesian()


#    @classmethod
#    def from_universe(cls, universe, inplace=False):
#        bases = universe.gaussian_basis_set[abs(universe.gaussian_basis_set['d']) > 0].groupby('set')
#        primdf = []
#        shfunc, func = -1, -1
#        for seht, x, y, z in zip(universe.atom['set'], universe.atom['x'],
#                                 universe.atom['y'], universe.atom['z']):
#            summ = universe.basis_set_summary.ix[seht]
#            b = bases.get_group(seht).groupby('shell_function')
#            for sh in range(len(b)):
#                prims = b.get_group(sh)
#                l = prims['l'].cat.as_ordered().max()
#                shfunc += 1
#                for l, m, n in enum_cartesian[l]:
#                    func += 1
#                    for alpha, d in zip(prims['alpha'].values, prims['d'].values):
#                        primdf.append([x, y, z, alpha, d, l, m, n, l + m + n, shfunc, func])
#        primdf = pd.DataFrame(primdf)
#        primdf.columns = ['xa', 'ya', 'za', 'alpha', 'd', 'l', 'm', 'n', 'L', 'shell', 'func']
#        if inplace:
#            universe.primitive = primdf
#        else:
#            return cls(primdf)

    @classmethod
    def from_universe(cls, universe, inplace=False):
        '''
        The minimum information specified by a basis set does not include
        expansion due to degeneracy from m_l. This will expand the basis in a
        systematic cartesian ordering convention to generate the full cartesian
        basis. The universe argument must already have a universe with atom,
        basis_set_summary, and gaussian_basis_set attributes.
        '''
        bases = universe.gaussian_basis_set[abs(universe.gaussian_basis_set['d']) > 0].groupby('set')
        primdf = []
        shfunc, func = -1, -1
        for seht, x, y, z in zip(universe.atom['set'], universe.atom['x'],
                                 universe.atom['y'], universe.atom['z']):
            summ = universe.basis_set_summary.ix[seht]
            b = bases.get_group(seht).groupby('shell_function')
            for sh in range(len(b)):
                prims = b.get_group(sh)
                l = prims['L'].cat.as_ordered().max()
                shfunc += 1
                for l, m, n in enum_cartesian[l]:
                    func += 1
                    for alpha, d in zip(prims['alpha'], prims['d']):
                        primdf.append([x, y, z, alpha, d, l, m, n, l + m + n, sh, shfunc, func, seht])
        primdf = pd.DataFrame(primdf)
        primdf.columns = ['xa', 'ya', 'za', 'alpha', 'd', 'l', 'm', 'n', 'L', 'shell_function', 'shell', 'func', 'set']
        if inplace:
            universe.primitive = primdf
        else:
            return cls(primdf)






class BasisSetOrder(BasisSet):
    '''
    BasisSetOrder uniquely determines the basis function ordering scheme for
    a given :class:`~exatomic.universe.Universe`. This table should be used
    if the ordering scheme is not programmatically available.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | basis_function    | int      | basis function index                      |
    +-------------------+----------+-------------------------------------------+
    | tag               | str      | symbolic atomic center                    |
    +-------------------+----------+-------------------------------------------+
    | center            | int      | numeric atomic center (1-based)           |
    +-------------------+----------+-------------------------------------------+
    | type              | str      | identifier equivalent to (l, ml)          |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['symbol', 'center', 'type']
    _indices = ['order']
    _categories = {'center': np.int64, 'type': str}

#class BasisSetMap(BasisSet):
#    '''
#    BasisSetMap provides the auxiliary information about relational mapping
#    between the complete uncontracted primitive basis set and the resultant
#    contracted basis set within an :class:`~exatomic.universe.Universe`.
#
#    +-------------------+----------+-------------------------------------------+
#    | Column            | Type     | Description                               |
#    +===================+==========+===========================================+
#    | tag               | str      | basis set identifier                      |
#    +-------------------+----------+-------------------------------------------+
#    | l                 | int      | oribtal angular momentum quantum number   |
#    +-------------------+----------+-------------------------------------------+
#    | nprim             | int      | number of primitives within shell         |
#    +-------------------+----------+-------------------------------------------+
#    | nbasis            | int      | number of basis functions within shell    |
#    +-------------------+----------+-------------------------------------------+
#    | degen             | bool     | False if cartesian, True if spherical     |
#    +-------------------+----------+-------------------------------------------+
#    '''
#    _columns = ['tag', 'nprim', 'nbasis', 'degen']
#    _indices = ['index']
#    #_categories = {'tag': str, 'shell': str, 'nbasis': np.int64, 'degen': bool}
#

class Overlap(DataFrame):
    '''
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
    | chi1              | int      | first basis function                      |
    +-------------------+----------+-------------------------------------------+
    | chi2              | int      | second basis function                     |
    +-------------------+----------+-------------------------------------------+
    | coefficient       | float    | overlap matrix element                    |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['chi1', 'chi2', 'coefficient', 'frame']
    _indices = ['index']

    def square(self, frame=0):
        nbas = np.round(np.roots([1, 1, -2 * self.shape[0]])[1]).astype(np.int64)
        tri = self[self['frame'] == frame].pivot('chi1', 'chi2', 'coefficient').fillna(value=0)
        return tri + tri.T - np.eye(nbas)

    @classmethod
    def from_square(cls, df):
        ndim = df.shape[0]
        arr = df.values
        arlen = ndim * (ndim + 1) // 2
        #chi1 = np.empty(arlen, dtype=np.int64)
        #chi2 = np.empty(arlen, dtype=np.int64)
        #coef = np.empty(arlen, dtype=np.float64)
        ret = np.empty((arlen,), dtype=[('chi1', 'i8'),
                                        ('chi2', 'i8'),
                                        ('coefficient', 'f8'),
                                        ('frame', 'i8')])
        cnt = 0
        for i in range(ndim):
            for j in range(i + 1):
                ret[cnt] = (i, j, arr[i, j], 0)
                cnt += 1
        return cls(ret)



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
    | frame             | int/cat  | non-unique integer                        |
    +-------------------+----------+-------------------------------------------+
    | x                 | int      | power of x                                |
    +-------------------+----------+-------------------------------------------+
    | y                 | int      | power of y                                |
    +-------------------+----------+-------------------------------------------+
    | z                 | int      | power of z                                |
    +-------------------+----------+-------------------------------------------+
    | l                 | int      | x + y + z                                 |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['l', 'x', 'y', 'z', 'frame']
    _indices = ['cart_order']
    _traits = ['l']
    _categories = {'l': np.int64, 'x': np.int64, 'y': np.int64, 'z': np.int64}

    def _custom_traits(self):
        #print(self.groupby('l').apply(lambda y: y['x'].values))
        #print(self.groupby('l').apply(lambda y: y['x'].values).to_json(orient='values'))
        #cgto_x = self.groupby('l').apply(lambda x: x['x'].values).to_json(orient='values')
        #cgto_x = Unicode(''.join(['[', cgto_x, ']'])).tag(sync=True)
        #cgto_y = self.groupby('l').apply(lambda x: x['y'].values).to_json(orient='values')
        #cgto_y = Unicode(''.join(['[', cgto_y, ']'])).tag(sync=True)
        #cgto_z = self.groupby('l').apply(lambda x: x['z'].values).to_json(orient='values')
        #cgto_z = Unicode(''.join(['[', cgto_z, ']'])).tag(sync=True)
        #return {'cartesiangtforder_x': cgto_x, 'cartesiangtforder_y': cgto_y,
        #        'cartesiangtforder_z': cgto_z}
        return {}

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
                          columns=['l', 'x', 'y', 'z'])
        df['frame'] = 0
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

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | frame             | int/cat  | non-unique integer                        |
    +-------------------+----------+-------------------------------------------+
    | l                 | int      | orbital angular momentum quantum number   |
    +-------------------+----------+-------------------------------------------+
    | ml                | int      | magnetic quantum number                   |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['l', 'ml', 'frame']
    _traits = ['l']
    _indices = ['spherical_order']

    def _custom_traits(self):
        sgto = self.groupby('frame').apply(lambda x: x.groupby('l').apply( lambda y: y['ml'].values))
        sgto = Unicode(sgto.to_json(orient='values')).tag(sync=True)
        return {'sphericalgtforder_ml': sgto}
        #Unicode('[' + self.groupby('l').apply(
        #        lambda x: x['ml'].values).to_json(orient='values') + ']').tag(sync=True)}

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
        df['frame'] = 0
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
