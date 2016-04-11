# -*- coding: utf-8 -*-
'''
Atom Dataframes
==========================
A dataframe containing the nuclear positions, forces, velocities, symbols, etc.
Examples of data that may exist in this dataframe are given below (note that
the dataframe is not limited to only these record types - rather this provides
a guide for what type of data is required and can be expected).

+-------------------+----------+-------------------------------------------+
| Column            | Type     | Description                               |
+===================+==========+===========================================+
| x                 | float    | position in x (req.)                      |
+-------------------+----------+-------------------------------------------+
| y                 | float    | position in y (req.)                      |
+-------------------+----------+-------------------------------------------+
| z                 | float    | position in z (req.)                      |
+-------------------+----------+-------------------------------------------+
| frame             | category | non-unique integer (req.)                 |
+-------------------+----------+-------------------------------------------+
| symbol            | category | element symbol (req.)                     |
+-------------------+----------+-------------------------------------------+
| fx                | float    | force in x                                |
+-------------------+----------+-------------------------------------------+
| fy                | float    | force in y                                |
+-------------------+----------+-------------------------------------------+
| fz                | float    | force in z                                |
+-------------------+----------+-------------------------------------------+
| vx                | float    | velocity in x                             |
+-------------------+----------+-------------------------------------------+
| vy                | float    | velocity in y                             |
+-------------------+----------+-------------------------------------------+
| vz                | float    | velocity in z                             |
+-------------------+----------+-------------------------------------------+
| label             | category | non-unique integer                        |
+-------------------+----------+-------------------------------------------+

See Also:
    :class:`~atomic.universe.Universe`
'''
import numpy as np
import pandas as pd
from traitlets import Dict, Unicode
from exa import DataFrame, SparseDataFrame
from exa.algorithms import supercell3
from atomic import Isotope


class BaseAtom(DataFrame):
    '''
    Base atom and related datframe.
    '''
    _indices = ['atom']
    _columns = ['x', 'y', 'z', 'symbol', 'frame']
    _traits = ['x', 'y', 'z']
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'label': np.int64, 'symbol': str}

    def _get_custom_traits(self):
        '''
        Creates four custom traits; radii, colors, symbols, and symbol codes.
        '''
        symbols = Unicode(self.groupby('frame').apply(
            lambda x: x['symbol'].cat.codes.values
        ).to_json(orient='values')).tag(sync=True)
        symmap = {i: v for i, v in enumerate(self['symbol'].cat.categories)}
        radii = Isotope.symbol_to_radius()[self['symbol'].unique()]
        radii = Dict({i: radii[v] for i, v in symmap.items()}).tag(sync=True)
        colors = Isotope.symbol_to_color()[self['symbol'].unique()]
        colors = Dict({i: colors[v] for i, v in symmap.items()}).tag(sync=True)
        return {'atom_symbols': symbols, 'atom_radii': radii,
                'atom_colors': colors}


class Atom(BaseAtom):
    '''
    Absolute positions of atoms and their symbol.
    '''
    def get_element_mass(self, inplace=False):
        '''
        Retrieve the mass of each element in the atom dataframe.
        '''
        masses = self['symbol'].astype('O').map(Isotope.symbol_to_mass())
        if inplace:
            self['mass'] = masses
        else:
            return masses

    def compute_simple_formula(self):
        '''
        Compute the simple formula for each frame.
        '''
        raise NotImplementedError()

    def _compute_unit_atom_static_cell(self, rxyz, oxyz):
        '''
        Given a static unit cell, compute the unit cell coordinates for each
        atom.

        Args:
            rxyz (:class:`~numpy.ndarray`): Unit cell magnitudes
            oxyz (:class:`~numpy.ndarray`): Unit cell origin

        Returns:
            sparse_df (:pandas:`~pandas.SparseDataFrame`): Sparse dataframe of in unit cell positions
        '''
        xyz = self[['x', 'y', 'z']]
        unit = np.mod(xyz, rxyz) + oxyz
        return UnitAtom(unit[unit != xyz].astype(np.float64).to_sparse())


class UnitAtom(SparseDataFrame):
    '''
    In unit cell coordinates (sparse) for periodic systems. These coordinates
    are used to update the corresponding :class:`~atomic.atom.Atom` object
    '''
    _indices = ['atom']
    _columns = ['x', 'y', 'z']


class ProjectedAtom(BaseAtom):
    '''
    Projected atom coordinates (e.g. on 3x3x3 supercell). These coordinates are
    typically associated with their corresponding indices in another dataframe.
    '''
    _indices = ['prjd_atom']
    _columns = ['x', 'y', 'z', 'symbol', 'frame', 'atom']
    _traits = ['x', 'y', 'z']
    _groupbys = ['frame']
    _categories = {'atom': np.int64, 'frame': np.int64, 'label': np.int64,
                   'symbol': str}


def compute_unit_atom(universe):
    '''
    Compute the unit cell coordinates of the atoms.

    Args:
        universe (:class:`~atomic.universe.Universe`): Atomic universe

    Returns:
        sparse_df (:pandas:`~pandas.SparseDataFrame`): Sparse dataframe of in unit cell positions
    '''
    if not universe.is_periodic:
        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
    if universe.is_variable_cell:
        raise NotImplementedError('Variable cell simulations not yet supported')
    idx = universe.frame.index[0]
    rxyz = universe.frame.ix[idx, ['rx', 'ry', 'rz']].values
    oxyz = universe.frame.ix[idx, ['ox', 'oy', 'oz']].values
    return universe.atom._compute_unit_atom_static_cell(rxyz, oxyz)


def compute_projected_atom(universe):
    '''
    Computes the 3x3x3 supercell coordinates from the unit cell coordinates.
    '''
    if not universe.is_periodic:
        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
    if universe.is_variable_cell:
        raise NotImplementedError('Variable cell simulations not yet supported')
    return _compute_projected_static(universe)


def _compute_projected_static(universe):
    '''
    Compute the 3x3x3 supercell coordinates given a static unit cell
    '''
    idx = universe.frame.index[0]
    x = universe.unit_atom['x'].values
    y = universe.unit_atom['y'].values
    z = universe.unit_atom['z'].values
    rx = universe.frame.ix[idx, 'rx']
    ry = universe.frame.ix[idx, 'ry']
    rz = universe.frame.ix[idx, 'rz']
    x, y, z = supercell3(x, y, z, rx, ry, rz)
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'z': z})
    df['frame'] = pd.Series(universe.atom['frame'].astype(np.int64).values.tolist() * 27, dtype='category')
    df['symbol'] = pd.Series(universe.atom['symbol'].astype(str).values.tolist() * 27, dtype='category')
    df['atom'] = pd.Series(universe.atom.index.values.tolist() * 27, dtype='category')
    #df['label'] = pd.Series(universe.atom['label'].astype(np.int64).values.tolist() * 27, dtype='category')
    return df
    #return ProjectedAtom(df)
