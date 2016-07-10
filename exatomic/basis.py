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
    For symbolic and discrete manipulations see :mod:`~exatomic.algorithms.basis`.
'''
import pandas as pd
import numpy as np
from collections import OrderedDict
from traitlets import Float, Int, Dict, Unicode
from exa import DataFrame
#from exatomic.algorithms.basis import spher_ml_count, cart_ml_count


lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'k': 7, 'l': 8,
        'm': 9, 'px': 1, 'py': 1, 'pz': 1}
spher_ml_count = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'k': 15,
                  'l': 17, 'm': 19}


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
    | alpha             | float    | value of :math:`\\alpha`                  |
    +-------------------+----------+-------------------------------------------+
    | d                 | float    | value of the contraction coefficient      |
    +-------------------+----------+-------------------------------------------+
    | shell_function    | int/cat  | basis function group identifier           |
    +-------------------+----------+-------------------------------------------+
    | l                 | int/cat  | orbital angular momentum quantum number   |
    +-------------------+----------+-------------------------------------------+
    | set               | int/cat  | index of unique basis set per unique atom |
    +-------------------+----------+-------------------------------------------+
    | frame             | int/cat  | non-unique integer                        |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['alpha', 'd', 'shell_function', 'l', 'set']
    _indices = ['primitive']
    _traits = ['shell_function']
    #_groupbys = ['frame']
    _precision = 8
    _categories = {'set': np.int64, 'l': np.int64, 'shell_function': np.int64,
                   'frame': np.int64}

    def _update_custom_traits(self):
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
             lambda z: z['l'].values).values)).to_json(orient='values')
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

    def _update_custom_traits(self):
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

    def _update_custom_traits(self):
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
